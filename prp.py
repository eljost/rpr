# [1] https://doi.org/10.1002/jcc.26495
#     Habershon, 2021

from functools import reduce
import itertools as it

import numpy as np

from pysisyphus.Geometry import Geometry
from pysisyphus.helpers import geom_loader
from pysisyphus.intcoords.setup import get_fragments, get_bond_sets
from pysisyphus.xyzloader import coords_to_trj


def precon(geom1, geom2, bonds_formed):
    # Place reactants at origin
    geom1.coords3d -= geom1.centroid[None, :]
    geom2.coords3d -= geom2.centroid[None, :]
    print("geom1.centroid", geom1.centroid)
    print("geom2.centroid", geom2.centroid)

    union = geom1 + geom2
    # union.jmol()


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


def get_molecular_radius(coords3d):
    coords3d = coords3d.copy()
    mean = coords3d.mean(axis=0)
    coords3d -= mean[None, :]
    distances = np.linalg.norm(coords3d, axis=1)
    std = max(0.9452, np.std(distances))  # at least 2 angstrom apart
    radius = distances.mean() + 2 * std
    return radius


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
    rfrag_num = len(rfrags)
    pfrag_num = len(pfrags)

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

    ###########
    # STAGE 1 #
    ###########

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

    ###########
    # STAGE 2 #
    ###########

    # Estimate fragment radii
    rradii = [get_molecular_radius(runion.coords3d[rfrag]) for rfrag in rfrag_lists]
    pradii = [get_molecular_radius(punion.coords3d[pfrag]) for pfrag in pfrag_lists]
    print(rradii, pradii)

    def hard_sphere_opt(
        frag_lists, radii, coords3d, kappa_s21=1.0, max_cycles=50, trj_fn=None
    ):
        coords3d = coords3d.copy()
        c3ds = list()
        for i in range(max_cycles):
            c3ds.append(coords3d.copy())
            for m, frag_m in enumerate(frag_lists):
                M_m = len(frag_m)
                h_m = radii[m]
                g_m = coords3d[frag_m].mean(axis=0)
                N_tot = 0.0
                Hs = list()
                for n, _ in enumerate(frag_lists):
                    if m == n:
                        continue
                    frag_n = frag_lists[n]
                    M_n = len(frag_n)
                    h_n = radii[n]
                    h_sum = h_m + h_n
                    g_n = coords3d[frag_n].mean(axis=0)
                    g_diff = g_m - g_n
                    g_dist = np.linalg.norm(g_diff)
                    H = 1 if g_dist <= h_sum else 0
                    Hs.append(H)
                    N_tot += 3 * M_m * H

                f_tot = np.zeros(3)
                for n, _ in enumerate(frag_lists):
                    if m == n:
                        continue
                    H = Hs.pop(0)
                    h_n = radii[n]
                    h_sum = h_m + h_n
                    g_n = coords3d[frag_n].mean(axis=0)
                    g_diff = g_m - g_n
                    g_dist = np.linalg.norm(g_diff)

                    phi_mn = kappa_s21 / N_tot * (g_dist - h_sum)
                    f_tot += phi_mn * H * g_diff / g_dist
                    f_tot_str = np.array2string(f_tot, precision=3)
                    print(
                        f"{i:02d}: H={H}, N={N_tot}, phi={phi_mn:.4f}, f_tot={f_tot_str}"
                    )
                coords3d[frag_m] -= f_tot[None, :]

        if trj_fn is not None:
            atoms = reactants.atoms
            coords_list = c3ds
            coords_to_trj(trj_fn, atoms, coords_list)
        return coords3d

    runion.coords3d = hard_sphere_opt(
        rfrag_lists, rradii, runion.coords3d, trj_fn="rstage2_opt.trj"
    )
    punion.coords3d = hard_sphere_opt(
        pfrag_lists, pradii, punion.coords3d, trj_fn="pstage2_opt.trj"
    )

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
        rot_coords = (runion.coords3d[rfrag]- gm).dot(rot_mat)
        # rot_coords += gm - rot_coords.mean(axis=0)
        # runion.coords3d[rfrag] = rot_coords
        runion.coords3d[rfrag] = rot_coords + gm - rot_coords.mean(axis=0)

    with open("rstage3.trj", "w") as handle:
        handle.write("\n".join((rstage3_pre_rot, runion.as_xyz(), products.as_xyz())))

    Ns = [0] * len(pfrag_lists)
    for (m, n), CPmn in CP.items():
        Ns[m] += len(CPmn)
    Ns2_tot = sum([N ** 2 for N in Ns])

    pstage3_pre_rot = punion.as_xyz()
    # Rotate P fragments
    for m, pfrag in enumerate(pfrag_lists):
        pc3d = punion.coords3d[pfrag]
        gm = pc3d.mean(axis=0)
        # r0Pm = pc3d - pc3d.mean(axis=0)[None, :]
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
