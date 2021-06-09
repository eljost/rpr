# [1] https://doi.org/10.1002/jcc.26495
#     Habershon, 2021

from functools import reduce

import numpy as np

from pysisyphus.calculators import HardSphereCalculator
from pysisyphus.Geometry import Geometry
from pysisyphus.helpers import geom_loader
from pysisyphus.intcoords.setup import get_fragments, get_bond_sets
from pysisyphus.xyzloader import coords_to_trj
from pysisyphus.optimizers.SteepestDescent import SteepestDescent


def get_fragments_and_bonds(geoms):
    if isinstance(geoms, Geometry) or len(geoms) == 1:
        geom = geoms
        atoms = geom.atoms
        coords3d = geom.coords3d
        bonds = [frozenset(bond) for bond in get_bond_sets(atoms, coords3d)]
        fragments = get_fragments(atoms, coords3d.flatten(), bond_inds=bonds)

        frag_bonds = [
            list(filter(lambda bond: bond <= frag, bonds)) for frag in fragments
        ]
        # frag_atoms = [[a for i, a in enumerate(atoms) if i in frag] for frag in fragments]

        # Assert that we do not have any interfragment bonds
        assert reduce((lambda x, y: x + len(y)), frag_bonds, 0) == len(bonds)
        union_geom = geom.copy(coord_type="cart")
    else:
        # Form union, determine consistent new indices for all atoms and calculate bonds
        raise Exception()

    # return fragments, frag_bonds, set(bonds), frag_atoms
    return fragments, frag_bonds, set(bonds), union_geom


def get_rot_mat(coords3d_1, coords3d_2, center=False):
    coords3d_1 = coords3d_1.copy().reshape(-1, 3)
    coords3d_2 = coords3d_2.copy().reshape(-1, 3)

    def _center(coords3d):
        return coords3d - coords3d.mean(axis=0)

    if center:
        coords3d_1 = _center(coords3d_1)
        coords3d_2 = _center(coords3d_2)

    tmp_mat = coords3d_1.T.dot(coords3d_2)
    U, W, Vt = np.linalg.svd(tmp_mat)
    rot_mat = U.dot(Vt)
    # Avoid reflections
    if np.linalg.det(rot_mat) < 0:
        U[:, -1] *= -1
        rot_mat = U.dot(Vt)
    return rot_mat


def get_trans_rot_vecs(coords3d, A, m, frag_lists, kappa=1):
    mfrag = frag_lists[m]
    mcoords3d = coords3d[mfrag]
    gm = mcoords3d.mean(axis=0)

    vt = np.zeros(3)
    vr = np.zeros(3)
    N = 0
    for n, nfrag in enumerate(frag_lists):
        if m == n:
            continue
        Amn = A[(m, n)]
        Anm = A[(n, m)]
        N += len(Amn) + len(Anm)
        for a in Amn:
            for b in Anm:
                rd = coords3d[b] - coords3d[a]
                gd = coords3d[a] - gm

                vt += abs(rd.dot(gd)) * rd / np.linalg.norm(rd)
                vr += np.cross(rd, gd)

    N *= 3 * len(mfrag)
    N_inv = 1 / N
    vt *= N_inv
    vr *= N_inv

    forces = kappa * (np.cross(-vr, mcoords3d - gm) + vt[None, :])
    return forces


def get_trans_rot_vecs2(
    mfrag,
    a_coords3d,
    b_coords3d,
    a_mats,
    b_mats,
    m,
    frag_lists,
    weight_func=None,
    skip=True,
    kappa=1,
):
    mcoords3d = a_coords3d[mfrag]
    gm = mcoords3d.mean(axis=0)

    if weight_func is None:

        def weight_func(m, n, a, b):
            return 1

    trans_vec = np.zeros(3)
    rot_vec = np.zeros(3)
    N = 0
    for n, nfrag in enumerate(frag_lists):
        if skip and (m == n):
            continue
        amn = a_mats[(m, n)]
        bnm = b_mats[(n, m)]
        N += len(amn) * len(bnm)
        for a in amn:
            for b in bnm:
                rd = b_coords3d[b] - a_coords3d[a]
                gd = a_coords3d[a] - gm
                weight = weight_func(a, b, m, n)

                trans_vec += weight * abs(rd.dot(gd)) * rd / np.linalg.norm(rd)
                rot_vec += weight * np.cross(rd, gd)

    N *= 3 * len(mfrag)
    N_inv = 1 / N
    trans_vec *= N_inv
    rot_vec *= N_inv

    forces = kappa * (np.cross(-rot_vec, mcoords3d - gm) + trans_vec[None, :])
    return forces


def sd_opt(geom, forces_getter, max_cycles=500, max_step=0.25, rms_thresh=0.005):
    coords = geom.coords.copy()
    for i in range(max_cycles):
        forces = forces_getter(coords)
        norm = np.linalg.norm(forces)
        if norm <= rms_thresh:
            print("Converged")
            break
        step = forces
        step *= min(max_step / np.abs(step).max(), 1)
        coords += step
    return coords


def precon_pos_orient(reactants, products):
    rfrags, rfrag_bonds, rbonds, runion = get_fragments_and_bonds(reactants)
    pfrags, pfrag_bonds, pbonds, punion = get_fragments_and_bonds(products)

    def get_which_frag(frags):
        which_frag = dict()
        for frag_ind, frag in enumerate(frags):
            which_frag.update({atom_ind: frag_ind for atom_ind in frag})
        return which_frag

    which_rfrag = get_which_frag(rfrags)
    which_pfrag = get_which_frag(pfrags)

    def center_fragments(frag_list, geom):
        c3d = geom.coords3d
        for frag in frag_list:
            mean = c3d[frag].mean(axis=0)
            c3d[frag] -= mean[None, :]

    rfrag_lists = [list(frag) for frag in rfrags]
    pfrag_lists = [list(frag) for frag in pfrags]

    pbond_diff = pbonds - rbonds  # Present in product(s)
    rbond_diff = rbonds - pbonds  # Present in reactant(s)

    def form_C(m_frags, n_frags):
        """Construct the C-matrices.

        Returns a dict with (m, n) keys, containing the respective
        unions of rectant fragment n and product fragment m.
        """
        C = dict()
        for m, m_frag in enumerate(m_frags):
            for n, n_frag in enumerate(n_frags):
                C[(m, n)] = list(m_frag & n_frag)
        return C

    CR = form_C(rfrags, pfrags)
    CP = {(n, m): union for (m, n), union in CR.items()}

    def form_B(C, bonds_formed):
        """Construct the B-matrices.

        Returns a dict with (m, n) keys, containing the respective
        subsets of C[(m, n)] that acutally participate in bond-breaking/forming.
        """
        B = dict()
        for (m, n), union in C.items():
            B[(m, n)] = set()
            for bond in bonds_formed:
                B[(m, n)] |= set(union) & bond
        B = {key: list(intersection) for key, intersection in B.items()}
        return B

    BR = form_B(CR, pbond_diff)
    BP = {(n, m): intersection for (m, n), intersection in BR.items()}

    def form_A(frags, which_frag, formed_bonds):
        """Construct the A-matrices.

        AR[(m, n)] (AP[(m, n)]) contains the subset of atoms in Rm (Pm) that forms
        bonds with Rn (Pn).
        """
        A = dict()
        for m, n in formed_bonds:
            key = (which_frag[m], which_frag[n])
            A.setdefault(key, list()).append(m)
            A.setdefault(key[::-1], list()).append(n)
        return A

    AR = form_A(rfrags, which_rfrag, pbond_diff)
    AP = form_A(pfrags, which_pfrag, rbond_diff)

    def form_G(frags, A):
        G = dict()
        for m, _ in enumerate(frags):
            for n, _ in enumerate(frags):
                if m == n:
                    continue
                G.setdefault(m, list()).extend(A[m, n])
        return G

    G = form_G(rfrags, AR)

    #########################################################
    # STAGE 1                                               #
    #                                                       #
    # Initial positioning of reactant and product molecules #
    #########################################################

    # Center fragments at their geometric average
    center_fragments(rfrag_lists, runion)
    center_fragments(pfrag_lists, punion)

    def get_steps_to_active_atom_mean(frag_lists, ind_dict, coords3d):
        frag_num = len(frag_lists)
        steps = np.zeros((frag_num, 3))
        for m, frag_m in enumerate(frag_lists):
            step_m = np.zeros(3)
            for n, _ in enumerate(frag_lists):
                if m == n:
                    continue
                active_inds = ind_dict[(n, m)]
                if len(active_inds) == 0:
                    continue
                step_m += coords3d[active_inds].mean(axis=0)
            step_m /= frag_num
            steps[m] = step_m
        return steps

    # Translate reactant molecules
    alphas = get_steps_to_active_atom_mean(rfrag_lists, AR, runion.coords3d)
    for rfrag, alpha in zip(rfrag_lists, alphas):
        runion.coords3d[rfrag] += alpha

    # Translate product molecules
    betas = get_steps_to_active_atom_mean(pfrag_lists, BR, punion.coords3d)
    sigmas = get_steps_to_active_atom_mean(pfrag_lists, CR, punion.coords3d)
    bs_half = (betas + sigmas) / 2
    for pfrag, bsh in zip(pfrag_lists, bs_half):
        punion.coords3d[pfrag] += bsh

    ##################################################
    # STAGE 2                                        #
    #                                                #
    # Intra-image Inter-molecular Hard-Sphere forces #
    ##################################################

    def hard_sphere_opt(geom, frag_lists, prefix):
        geom_ = geom.copy()
        calc = HardSphereCalculator(geom_, frag_lists)
        geom_.set_calculator(calc)
        opt_kwargs = {
            "max_cycles": 500,
            "max_step": 0.5,
            # "dump": True,
            "prefix": prefix,
        }
        opt = SteepestDescent(geom_, **opt_kwargs)
        opt.run()

        return geom_.coords3d

    runion.coords3d = hard_sphere_opt(runion, rfrag_lists, "R")
    punion.coords3d = hard_sphere_opt(punion, pfrag_lists, "P")

    ####################################
    # STAGE 3                          #
    #                                  #
    # Initial orientation of molecules #
    ####################################

    # Rotate R fragments
    alphas = get_steps_to_active_atom_mean(rfrag_lists, AR, runion.coords3d)
    gammas = np.zeros_like(alphas)
    for m, rfrag in enumerate(rfrag_lists):
        Gm = G[m]
        gammas[m] = runion.coords3d[Gm].mean(axis=0)
    r_means = np.array([runion.coords3d[frag].mean(axis=0) for frag in rfrag_lists])

    rstage3_pre_rot = runion.as_xyz()
    for m, rfrag in enumerate(rfrag_lists):
        gm = r_means[m]
        rot_mat = get_rot_mat(gammas[m] - gm, alphas[m] - gm)
        rot_coords = (runion.coords3d[rfrag] - gm).dot(rot_mat)
        runion.coords3d[rfrag] = rot_coords + gm - rot_coords.mean(axis=0)

    with open("rstage3.trj", "w") as handle:
        handle.write("\n".join((rstage3_pre_rot, runion.as_xyz(), products.as_xyz())))

    Ns = [0] * len(pfrag_lists)
    for (m, n), CPmn in CP.items():
        Ns[m] += len(CPmn)

    # Rotate P fragments
    pstage3_pre_rot = punion.as_xyz()
    for m, pfrag in enumerate(pfrag_lists):
        pc3d = punion.coords3d[pfrag]
        gm = pc3d.mean(axis=0)
        r0Pm = pc3d - gm[None, :]
        mu_Pm = np.zeros_like(r0Pm)
        N = Ns[m]
        for n, rfrag in enumerate(rfrag_lists):
            CPmn = CP[(m, n)]
            RPmRn = get_rot_mat(
                punion.coords3d[CPmn], runion.coords3d[CPmn], center=True
            )
            print(f"m={m}, n={n}, len(CPmn)={len(CPmn)}, rot_mat={rot_mat.shape}")
            # Eq. (A2) in [1]
            r0Pmn = np.einsum("ij,jk->ki", RPmRn, r0Pm.T)
            mu_Pm += len(CPmn) ** 2 / N * r0Pmn
        rot_mat = get_rot_mat(r0Pm, mu_Pm, center=True)
        rot_coords = r0Pm.dot(rot_mat)
        punion.coords3d[pfrag] = rot_coords + gm - rot_coords.mean(axis=0)

    with open("pstage3.trj", "w") as handle:
        handle.write("\n".join((pstage3_pre_rot, punion.as_xyz(), products.as_xyz())))

    """
    STAGE 4
    Alignment of reactive atoms


    This stage involves three forces: hard-sphere forces and two kinds
    of average translational (^t) and rotational (^r) forces (v and w,
    (A3) - (A5) in [1]).

    v^t and v^r arise from atoms in A^Rnm and A^Rmn, that is atoms that
    participate in bond forming/breaking in R. The translational force
    is usually attractive, which is counteracted by the repulsive hard-sphere
    forces.
    """

    def weight_func(m, n, a, b):
        """As required for (A5) in [1]."""
        try:
            return 1 if a in BR[(m, n)] else 0.5
        except KeyError:
            return 0.5

    s4_coords = list()
    rhs_calc = HardSphereCalculator(runion, rfrag_lists, kappa=50)
    # def r_forces_getter(coords):
    for i in range(500):
        s4_coords.append(runion.coords3d.copy())
        forces = np.zeros_like(runion.coords3d)
        for m, mfrag in enumerate(rfrag_lists):
            # forces = get_trans_rot_vecs(runion.coords3d, AR, m, rfrag_lists)
            v_forces = get_trans_rot_vecs2(
                mfrag, runion.coords3d, runion.coords3d, AR, AR, m, rfrag_lists
            )
            # np.testing.assert_allclose(v_forces, forces)
            w_forces = get_trans_rot_vecs2(
                mfrag,
                runion.coords3d,
                punion.coords3d,
                CR,
                CP,
                m,
                pfrag_lists,
                weight_func=weight_func,
                skip=False,
            )
            hs_res = rhs_calc.get_forces(runion.atoms, runion.coords)
            # import pdb; pdb.set_trace()
            hs_forces = hs_res["forces"].reshape(-1, 3)
            # print(f"norm(w_forces)={np.linalg.norm(w_forces):.4f}")
            forces[mfrag] = v_forces + w_forces + hs_forces[mfrag]
            # forces[mfrag] = v_forces + w_forces# + hs_forces[mfrag]
        step = forces
        norm = np.linalg.norm(step)
        if norm < 0.5:
            print("Converged")
            break
        max_step = 0.2
        if norm > max_step:
            step = max_step * step / norm
        new_coords3d = runion.coords3d + step
        print(f"{i:02d}: norm={norm:.4f}")
        runion.coords3d = new_coords3d

    atoms = runion.atoms
    coords_list = s4_coords
    coords_to_trj("rs4.trj", atoms, coords_list)


def run():
    # Patch covalent radii, so we can fome some bonds
    from pysisyphus.elem_data import COVALENT_RADII as CR

    CR["q"] = 1.5
    # educt, ts, product = geom_loader("00_c2no2.trj")
    # educt, product = geom_loader("test.trj")
    # educt, product = geom_loader("figure1.trj")
    # educt, product = geom_loader("fig2.trj")
    educt, product = geom_loader("fig2_mod.trj")
    precon_pos_orient(educt, product)


if __name__ == "__main__":
    run()
