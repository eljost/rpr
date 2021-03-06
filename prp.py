# [1] https://doi.org/10.1002/jcc.26495
#     Habershon, 2021

from functools import reduce
from pprint import pprint

import numpy as np

from pysisyphus.calculators import (
    HardSphere,
    TransTorque,
    AtomAtomTransTorque,
    Composite,
)
from pysisyphus.constants import BOHR2ANG
from pysisyphus.Geometry import Geometry
from pysisyphus.helpers import geom_loader, align_coords
from pysisyphus.intcoords.setup import get_fragments, get_bond_sets
from pysisyphus.xyzloader import coords_to_trj, make_xyz_str

# from pysisyphus.optimizers.SteepestDescent import SteepestDescent


class SteepestDescent:
    def __init__(
        self,
        geom,
        max_cycles=1000,
        max_step=0.05,
        rms_force=0.05,
        rms_force_only=True,
        prefix=None,
        dump=False,
        print_mod=25,
    ):
        self.geom = geom
        self.max_cycles = max_cycles
        self.max_step = max_step
        self.rms_force = rms_force
        self.rms_force_only = rms_force_only
        self.prefix = prefix
        self.dump = dump
        self.print_mod = print_mod

        self.all_coords = np.zeros((max_cycles, self.geom.coords.size))

    def run(self):
        coords = self.geom.coords.copy()

        for i in range(self.max_cycles):
            self.all_coords[i] = coords.copy()
            results = self.geom.get_energy_and_forces_at(coords)
            forces = results["forces"]
            # forces = forces_getter(coords)
            norm = np.linalg.norm(forces)
            rms = np.sqrt(np.mean(forces ** 2))
            if rms <= self.rms_force:
                print(f"Converged in cycle {i}. Breaking.")
                break
            step = forces.copy()
            step *= min(self.max_step / np.abs(step).max(), 1)
            if i % self.print_mod == 0:
                print(
                    f"{i:03d}: |forces|={norm: >12.6f} "
                    f"rms(forces)={np.sqrt(np.mean(forces**2)): >12.6f} "
                    f"|step|={np.linalg.norm(step): >12.6f}"
                )
            coords += step
        self.geom.coords = coords
        self.all_coords = self.all_coords[: i + 1]


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


def sd_opt(geom, forces_getter, max_cycles=1000, max_step=0.05, rms_thresh=0.05):
    coords = geom.coords.copy()
    all_coords = np.zeros((max_cycles, coords.size))

    for i in range(max_cycles):
        all_coords[i] = coords.copy()
        forces = forces_getter(coords)
        norm = np.linalg.norm(forces)
        rms = np.sqrt(np.mean(forces ** 2))
        if rms <= rms_thresh:
            print(f"Converged in cycle {i}. Breaking.")
            break
        step = forces.copy()
        step *= min(max_step / np.abs(step).max(), 1)
        if i % 25 == 0:
            print(
                f"{i:03d}: |forces|={norm: >12.6f} rms(forces)={np.sqrt(np.mean(forces**2)): >12.6f} "
                f"|step|={np.linalg.norm(step): >12.6f}"
            )
        coords += step

    return coords, all_coords[: i + 1]


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


def report_frags(rgeom, pgeom, rfrags, pfrags, rbond_diff, pbond_diff):
    for name, geom in (("Reactant(s)", rgeom), ("Product(s)", pgeom)):
        print(f"{name}: {geom}\n\n{geom.as_xyz()}\n")

    atoms = rgeom.atoms

    def get_frag_atoms(geom, frag):
        atoms = geom.atoms
        return [atoms[i] for i in frag]

    for name, geom, frags in (("reactant", rgeom, rfrags), ("product", pgeom, pfrags)):
        print(f"{len(frags)} fragment(s) in {name} image\n")
        for frag in frags:
            frag_atoms = get_frag_atoms(geom, frag)
            frag_coords = geom.coords3d[list(frag)]
            frag_xyz = make_xyz_str(frag_atoms, frag_coords * BOHR2ANG)
            print(frag_xyz + "\n")

    def print_bonds(geom, bonds):
        for from_, to_ in bonds:
            from_atom, to_atom = [geom.atoms[i] for i in (from_, to_)]
            print(f"\t({from_: >3d}{from_atom} - {to_: >3d}{to_atom})")

    print("Bonds broken in reactant image:")
    print_bonds(rgeom, rbond_diff)
    print()
    print("Bonds formed in product image:")
    print_bonds(pgeom, pbond_diff)
    print()


def report_mats(name, mats):
    for (m, n), indices in mats.items():
        print(f"{name}({m}, {n}): {indices}")
    print()


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

    report_frags(runion, punion, rfrags, pfrags, rbond_diff, pbond_diff)

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
    print("CR(m, n), subset of atoms in molecule Rn which are in Pm after reaction.")
    report_mats("CR", CR)
    print("CP(m, n), subset of atoms in molecule Pn which are in Rm before reaction.")
    report_mats("CP", CP)

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
    print("BR(m, n), subset of atoms in CRnm actually involved in bond forming/breaking.")
    report_mats("BR", BR)
    print("BP(m, n), subset of atoms in CPnm actually involved in bond forming/breaking.")
    report_mats("BP", BP)

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
    print(f"AR(m, n), subset of atoms in Rm that form bonds to atoms in Rn.")
    report_mats("AR", AR)
    print("AP(m, n), subset of atoms in Pm which had bonds with Pn (formerly bonded in R).")
    report_mats("AP", AP)

    def form_G(frags, A):
        G = dict()
        for m, _ in enumerate(frags):
            for n, _ in enumerate(frags):
                if m == n:
                    continue
                G.setdefault(m, list()).extend(A[m, n])
        return G

    G = form_G(rfrags, AR)
    print(f"G: {G}")

    import sys; sys.exit()

    # Initial, centered, coordinates and 5 stages
    r_coords = np.zeros((6, runion.coords.size))
    p_coords = np.zeros((6, punion.coords.size))

    def backup_coords(stage):
        assert 0 <= stage < 6
        r_coords[stage] = runion.coords.copy()
        p_coords[stage] = punion.coords.copy()

    #########################################################
    # STAGE 1                                               #
    #                                                       #
    # Initial positioning of reactant and product molecules #
    #########################################################

    # Center fragments at their geometric average
    center_fragments(rfrag_lists, runion)
    center_fragments(pfrag_lists, punion)
    backup_coords(0)

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

    backup_coords(1)

    ##################################################
    # STAGE 2                                        #
    #                                                #
    # Intra-image Inter-molecular Hard-Sphere forces #
    ##################################################

    def stage2_hard_sphere_opt(geom, frag_lists, prefix):
        geom_ = geom.copy()
        calc = HardSphere(geom_, frag_lists)
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

    runion.coords3d = stage2_hard_sphere_opt(runion, rfrag_lists, "R")
    punion.coords3d = stage2_hard_sphere_opt(punion, pfrag_lists, "P")

    backup_coords(2)

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

    # with open("rstage3.trj", "w") as handle:
    # handle.write("\n".join((rstage3_pre_rot, runion.as_xyz(), products.as_xyz())))

    Ns = [0] * len(pfrag_lists)
    for (m, n), CPmn in CP.items():
        Ns[m] += len(CPmn)

    # Rotate P fragments
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

    # with open("pstage3.trj", "w") as handle:
    # handle.write("\n".join((pstage3_pre_rot, punion.as_xyz(), products.as_xyz())))

    backup_coords(3)

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

    def stage4_opt(geom, keys_calcs):
        calc = Composite("hardsphere + v + w", keys_calcs=keys_calcs)
        geom.set_calculator(calc)
        opt_kwargs = {
            "max_step": 0.05,
            "max_cycles": 1000,
            "rms_force": 0.05,
            "rms_force_only": True,
        }
        opt = SteepestDescent(geom, **opt_kwargs)
        opt.run()

    def r_weight_func(m, n, a, b):
        """As required for (A5) in [1]."""
        return 1 if a in BR[(m, n)] else 0.5

    vr_trans_torque = TransTorque(rfrag_lists, rfrag_lists, AR, AR)
    wr_trans_torque = TransTorque(
        rfrag_lists,
        pfrag_lists,
        CR,
        CP,
        weight_func=r_weight_func,
        skip=False,
        b_coords3d=punion.coords3d,
    )
    r_keys_calcs = {
        "hardsphere": HardSphere(runion, rfrag_lists, kappa=50),
        "v": vr_trans_torque,
        "w": wr_trans_torque,
    }
    stage4_opt(runion, r_keys_calcs)

    def p_weight_func(m, n, a, b):
        """As required for (A5) in [1]."""
        return 1 if a in BP[(m, n)] else 0.5

    vp_trans_torque = TransTorque(pfrag_lists, pfrag_lists, AP, AP)
    wp_trans_torque = TransTorque(
        pfrag_lists,
        rfrag_lists,
        CP,
        CR,
        weight_func=p_weight_func,
        skip=False,
        b_coords3d=runion.coords3d,
    )
    p_keys_calcs = {
        "hardsphere": HardSphere(punion, pfrag_lists, kappa=50),
        "v": vp_trans_torque,
        "w": wp_trans_torque,
    }
    stage4_opt(punion, p_keys_calcs)

    backup_coords(4)

    """
    STAGE 5
    Refinement of atomic positions using further hard-sphere forces.
    """

    def stage5_opt(geom, keys_calcs):
        # calc = Composite("hardsphere + v + w + z", keys_calcs=keys_calcs)
        # Hardsphere between molecules that do not interact is missing
        calc = Composite("v + w + z", keys_calcs=keys_calcs)
        geom.set_calculator(calc)
        opt_kwargs = {
            "max_step": 0.05,
            "max_cycles": 1000,
            "rms_force": 0.05,
            "rms_force_only": True,
            "dump": True,
            "print_mod": 5,
        }
        opt = SteepestDescent(geom, **opt_kwargs)
        opt.run()

    def r_weight_func(m, n, a, b):
        """As required for (A5) in [1]."""
        return 1 if a in BR[(m, n)] else 0.5

    vr_trans_torque = TransTorque(rfrag_lists, rfrag_lists, AR, AR, kappa=1.0)
    wr_trans_torque = TransTorque(
        rfrag_lists,
        pfrag_lists,
        CR,
        CP,
        weight_func=r_weight_func,
        skip=False,
        b_coords3d=punion.coords3d,
        kappa=3.0,
    )
    zr_aa_trans_torque = AtomAtomTransTorque(runion, rfrag_lists, AR)
    r_keys_calcs = {
        # "hardsphere": HardSphere(runion, rfrag_lists, kappa=50),
        "v": vr_trans_torque,
        "w": wr_trans_torque,
        "z": zr_aa_trans_torque,
    }
    stage5_opt(runion, r_keys_calcs)

    backup_coords(5)

    with open("rb5.xyz", "w") as handle:
        handle.write(runion.as_xyz())
    import pickle

    with open("rb5", "wb") as handle:
        pickle.dump((rfrag_lists, AR), handle)

    def dump_stages(fn, atoms, coords_list):
        align_coords(coords_list)
        comments = [f"Stage {i}" for i in range(coords_list.shape[0])]
        coords_to_trj(fn, atoms, coords_list, comments=comments)

    dump_stages("r_coords.trj", runion.atoms, r_coords)
    dump_stages("p_coords.trj", punion.atoms, p_coords)


def stage5():
    runion = geom_loader("rb5.xyz")
    import pickle

    with open("rb5", "rb") as handle:
        rfrag_lists, AR = pickle.load(handle)

    raa = AtomAtomTransTorque(runion, rfrag_lists, AR)
    res = raa.get_forces(runion.atoms, runion.coords)
    forces = res["forces"]


def stage5_comp():
    runion = geom_loader("stage5comp.xyz")
    # runion.jmol()

    rfrag_lists = ([0, 1], [2, 3])
    AR = {
        (0, 1): [1],
        (1, 0): [2],
    }

    raa = AtomAtomTransTorque(runion, rfrag_lists, AR)
    # zt, zr = raa.get_forces(runion.atoms, runion.coords)
    # ztr, zrr = raa.get_forces_naive(runion.atoms, runion.coords)
    # np.testing.assert_allclose(zt, ztr)
    # np.testing.assert_allclose(zr, zrr)
    # print("MATCH!")
    # print("zt", zt)
    # print("zr", zr)
    res = raa.get_forces(runion.atoms, runion.coords)
    res_ref = raa.get_forces_naive(runion.atoms, runion.coords)
    np.testing.assert_allclose(res["forces"], res_ref["forces"])
    print("MATCH!")


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
    # stage5()
    # stage5_comp()
