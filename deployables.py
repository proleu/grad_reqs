# Python standard library
from typing import Iterator, List, Optional

from pyrosetta.distributed import requires_init
from pyrosetta.distributed.packed_pose.core import PackedPose

# 3rd party library imports
# Rosetta library imports
from pyrosetta.rosetta.core.pose import Pose

# Custom library imports


def get_fullatom_rmsd(design: Pose, prediction: Pose) -> float:
    """
    :param: design: Pose to compare to prediction
    :param: prediction: Pose to compare to design
    :return: RMSD between the two poses, calculated over all atoms
    Calculate fullatom RMSD of prediction to design.
    """
    import pyrosetta

    rmsd_calc = pyrosetta.rosetta.core.simple_metrics.metrics.RMSDMetric()
    rmsd_calc.set_rmsd_type(pyrosetta.rosetta.core.scoring.rmsd_atoms(7))
    rmsd_calc.set_run_superimpose(True)
    rmsd_calc.set_comparison_pose(design)
    rmsd = float(rmsd_calc.calculate(prediction))
    return rmsd


@requires_init
def pack_around_ligand(
    packed_pose_in: Optional[PackedPose] = None, **kwargs
) -> Iterator[PackedPose]:
    """
    :param: packed_pose_in: PackedPose object to pack around ligand.
    :param: kwargs: Keyword arguments to pass to pack_around_ligand.
    :return: Iterator of PackedPose objects.
    Pack residues contacting ligand.
    """

    import sys
    from pathlib import Path
    from time import time

    import pyrosetta
    import pyrosetta.distributed.io as io
    from pyrosetta.rosetta.core.select.residue_selector import (
        AndResidueSelector,
        ChainSelector,
    )

    # insert the root of the repo into the sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from crispy_shifty.protocols.cleaning import path_to_pose_or_ppose
    from crispy_shifty.protocols.design import (
        gen_scorefxn,
        gen_std_layer_design,
        gen_task_factory,
        interface_between_selectors,
        pack_rotamers,
    )
    from crispy_shifty.utils.io import print_timestamp

    start_time = time()

    # generate poses or convert input packed pose into pose
    if packed_pose_in is not None:
        poses = [io.to_pose(packed_pose_in)]
        pdb_path = "none"
    else:
        pdb_path = kwargs["pdb_path"]
        poses = path_to_pose_or_ppose(
            path=pdb_path, cluster_scores=True, pack_result=False
        )

    print_timestamp("Setting design options", start_time)

    # generate interface and chA selectors
    chA = ChainSelector("1")
    chB = ChainSelector("2")
    interface = interface_between_selectors(chA, chB)
    interface_and_chA = AndResidueSelector(interface, chA)
    # generate scorefxn
    scorefxn = gen_scorefxn()
    # generate task factory
    task_factory = gen_task_factory(
        design_sel=interface_and_chA,
        pack_nbhd=True,
        extra_rotamers_level=1,
        limit_arochi=True,
        restrict_pro_gly=True,
        layer_design=gen_std_layer_design(),
    )

    for pose in poses:
        print_timestamp("Packing around ligand", start_time)
        pose.update_residue_neighbors()
        scores = dict(pose.scores)
        original_pose = pose.clone()
        pack_rotamers(pose, task_factory, scorefxn)
        print_timestamp("Packing complete", start_time)
        pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, "path_in", pdb_path)
        ppose = io.to_packed(pose)
        yield ppose


@requires_init
def mpnn_around_ligand(
    packed_pose_in: Optional[PackedPose] = None, **kwargs
) -> Iterator[PackedPose]:
    """
    :param: packed_pose_in: a PackedPose object to be interface designed with MPNN.
    :param: kwargs: keyword arguments to be passed to MPNNDesign, or this function.
    :return: an iterator of PackedPose objects.
    """

    import sys
    from itertools import product
    from pathlib import Path
    from time import time

    import pyrosetta
    import pyrosetta.distributed.io as io
    from pyrosetta.rosetta.core.select.residue_selector import (
        AndResidueSelector,
        ChainSelector,
        NotResidueSelector,
        ResidueIndexSelector,
    )

    # insert the root of the repo into the sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from crispy_shifty.protocols.cleaning import path_to_pose_or_ppose
    from crispy_shifty.protocols.design import interface_between_selectors
    from crispy_shifty.protocols.mpnn import MPNNDesign, MPNNLigandDesign
    from crispy_shifty.utils.io import print_timestamp

    start_time = time()

    # generate poses or convert input packed pose into pose
    if packed_pose_in is not None:
        poses = [io.to_pose(packed_pose_in)]
        pdb_path = "none"
    else:
        pdb_path = kwargs["pdb_path"]
        poses = path_to_pose_or_ppose(
            path=pdb_path, cluster_scores=True, pack_result=False
        )
    # try to get ligand_params from kwargs
    try:
        params_path = kwargs["ligand_params"]
    except KeyError:
        raise KeyError("No ligand_params provided.")

    # setup dict for MPNN design areas
    print_timestamp("Setting up design selectors", start_time)
    # make a designable residue selector of only the interface residues
    chA = ChainSelector(1)
    chB = ChainSelector(2)
    not_interface_selector = NotResidueSelector(interface_between_selectors(chA, chB))
    chA_not_interface = AndResidueSelector(chA, not_interface_selector)

    for pose in poses:
        pose.update_residue_neighbors()
        scores = dict(pose.scores)
        original_pose = pose.clone()
        # iterate over the mpnn parameter combinations
        print_timestamp("Designing interface with MPNN", start_time)
        designed_poses = {}
        # get indices of non-interface residues
        non_interface_indices = [
            str(i)
            for i, in_sel in enumerate(chA_not_interface.apply(pose), start=1)
            if in_sel
        ]
        # delete chB
        sc = pyrosetta.rosetta.protocols.simple_moves.SwitchChainOrderMover()
        sc.chain_order("1")
        sc.apply(pose)
        mpnn_design_area = ResidueIndexSelector(",".join(non_interface_indices))
        # construct the MPNNDesign object
        mpnn_design = MPNNDesign(
            design_selector=mpnn_design_area,
            omit_AAs="X",
            **kwargs,
        )
        # design the pose
        mpnn_design.apply(pose)
        # put chB back
        chB_pose = original_pose.split_by_chain(2)
        pyrosetta.rosetta.core.pose.append_pose_to_pose(pose, chB_pose, new_chain=True)
        sc.chain_order("12")
        vanilla_scores = dict(pose.scores)
        sc.apply(pose)
        # put the scores back
        for key, value in vanilla_scores.items():
            pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, key, value)
        designed_poses["vanilla"] = pose
        # now do ligandmpnn
        pose = original_pose.clone()
        # construct the MPNNLigandDesign object
        mpnn_design = MPNNLigandDesign(
            design_selector=chA,
            params=params_path,
            omit_AAs="CX",
            **kwargs,
        )
        # design the pose
        mpnn_design.apply(pose)
        designed_poses["ligand"] = pose
        print_timestamp("MPNN design complete, updating pose datacache", start_time)
        # update the poses with the updated scores dict
        for design_method, designed_pose in designed_poses.items():
            final_scores = {**scores, **dict(designed_pose.scores)}
            pyrosetta.rosetta.core.pose.setPoseExtraScore(
                designed_pose, "type", design_method
            )
            for key, value in final_scores.items():
                pyrosetta.rosetta.core.pose.setPoseExtraScore(designed_pose, key, value)
            yield designed_pose


@requires_init
def fold_binder(
    packed_pose_in: Optional[PackedPose] = None, **kwargs
) -> Iterator[PackedPose]:
    """
    :param: packed_pose_in: a PackedPose object to fold with the superfold script.
    :param: kwargs: keyword arguments to be passed to the superfold script.
    :return: an iterator of PackedPose objects.
    """

    import sys
    from operator import gt, lt
    from pathlib import Path
    from time import time

    import pyrosetta
    import pyrosetta.distributed.io as io
    from pyrosetta.rosetta.core.pose import Pose

    # insert the root of the repo into the sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from crispy_shifty.protocols.cleaning import path_to_pose_or_ppose
    from crispy_shifty.protocols.folding import (
        SuperfoldRunner,
        generate_decoys_from_pose,
    )
    from crispy_shifty.protocols.mpnn import dict_to_fasta, fasta_to_dict
    from crispy_shifty.protocols.states import range_CA_align
    from crispy_shifty.utils.io import cmd, print_timestamp

    start_time = time()
    # hacky split pdb_path into pdb_path and fasta_path
    pdb_path = kwargs.pop("pdb_path")
    pdb_path, fasta_path = tuple(pdb_path.split("____"))

    # generate poses or convert input packed pose into pose
    if packed_pose_in is not None:
        poses = [io.to_pose(packed_pose_in)]
        pdb_path = "none"
    else:
        # skip the kwargs check
        poses = path_to_pose_or_ppose(
            path=pdb_path, cluster_scores=True, pack_result=False
        )

    for pose in poses:
        pose.update_residue_neighbors()
        scores = dict(pose.scores)
        # change name of MPNN design type
        design_type = scores["type"]
        del scores["type"]
        scores["design_type"] = design_type
        original_pose = pose.clone()
        # keep a copy of chB for later
        chB_pose = original_pose.split_by_chain(2)
        # delete chain B
        sc = pyrosetta.rosetta.protocols.simple_moves.SwitchChainOrderMover()
        sc.chain_order("1")
        sc.apply(pose)
        print_timestamp("Setting up for AF2", start_time)
        runner = SuperfoldRunner(
            pose=pose, fasta_path=fasta_path, load_decoys=True, **kwargs
        )
        runner.setup_runner(file=fasta_path)
        # reference_pdb is the tmp.pdb
        reference_pdb = str(Path(runner.get_tmpdir()) / "tmp.pdb")
        flag_update = {
            "--reference_pdb": reference_pdb,
        }
        runner.update_flags(flag_update)
        runner.update_command()
        print_timestamp("Running AF2", start_time)
        runner.apply(pose)
        print_timestamp("AF2 complete, updating pose datacache", start_time)
        # update the scores dict
        scores.update(pose.scores)
        # update the pose with the updated scores dict
        for key, value in scores.items():
            pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, key, value)
        # setup prefix, rank_on, filter_dict (in this case we can't get from kwargs)
        filter_dict = {
            "mean_plddt": (gt, 85.0),
            "rmsd_to_reference": (lt, 1.50),
        }
        rank_on = "mean_plddt"
        prefix = "mpnn_seq"
        print_timestamp("Generating decoys", start_time)
        # get sc ready for putting back chB and rebuilding PDBInfo
        sc.chain_order("12")
        for decoy in generate_decoys_from_pose(
            pose,
            filter_dict=filter_dict,
            generate_prediction_decoys=True,
            label_first=True,
            prefix=prefix,
            rank_on=rank_on,
        ):
            # get decoy scores
            final_scores = dict(decoy.scores)
            scores.update(final_scores)
            # align decoy back to original pose
            range_CA_align(
                decoy,
                original_pose,
                decoy.chain_begin(1),
                decoy.chain_end(1),
                original_pose.chain_begin(1),
                original_pose.chain_end(1),
            )
            # append chain B to decoy
            pyrosetta.rosetta.core.pose.append_pose_to_pose(
                decoy, chB_pose, new_chain=True
            )
            # rebuild PDBInfo
            sc.apply(decoy)
            # add scores back into decoy
            for key, value in scores.items():
                if "mpnn_seq" not in key:
                    pyrosetta.rosetta.core.pose.setPoseExtraScore(decoy, key, value)
                else:
                    continue
            # add FA_rmsd to scores after computing it
            fa_rmsd = get_fullatom_rmsd(original_pose, decoy)
            pyrosetta.rosetta.core.pose.setPoseExtraScore(decoy, "fa_rmsd", fa_rmsd)
            packed_decoy = io.to_packed(decoy)
            yield packed_decoy


@requires_init
def redock_ligand(
    packed_pose_in: Optional[PackedPose] = None, **kwargs
) -> Iterator[PackedPose]:
    """
    :param: packed_pose_in: PackedPose object to redock.
    :param: kwargs: Keyword arguments to pass to redock_ligand.
    :return: Iterator of PackedPose objects.
    Cartesian minimization of the AF2 decoy to relax into the Rosetta sfxn, then
    GALigandDock to dock the ligand into the relaxed decoy structure.
    """

    import sys
    from pathlib import Path
    from time import time

    import pyrosetta
    import pyrosetta.distributed.io as io
    from pyrosetta.rosetta.core.select.residue_selector import (
        AndResidueSelector,
        ChainSelector,
    )

    # insert the root of the repo into the sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from crispy_shifty.protocols.cleaning import path_to_pose_or_ppose
    from crispy_shifty.protocols.design import (
        fast_relax,
        gen_movemap,
        gen_scorefxn,
        gen_task_factory,
        interface_between_selectors,
    )
    from crispy_shifty.utils.io import print_timestamp

    start_time = time()

    # generate poses or convert input packed pose into pose
    if packed_pose_in is not None:
        poses = [io.to_pose(packed_pose_in)]
        pdb_path = "none"
    else:
        pdb_path = kwargs["pdb_path"]
        poses = path_to_pose_or_ppose(
            path=pdb_path, cluster_scores=True, pack_result=False
        )

    print_timestamp("Setting relax options", start_time)

    # check for custom_relax_script kwarg
    if "custom_relax_script" in kwargs:
        relax_script = kwargs["custom_relax_script"]
    else:
        relax_script = "MonomerRelax2019"

    # generate interface and chA selectors
    chA = ChainSelector("1")
    chB = ChainSelector("2")
    interface = interface_between_selectors(chA, chB)
    interface_and_chA = AndResidueSelector(interface, chA)
    # generate cartesian constrained scorefxn
    scorefxn = gen_scorefxn(cartesian=True, weights="beta_cst.wts")
    # generate task factory
    task_factory = gen_task_factory(
        pack_sel=chA,
        extra_rotamers_level=2,
        ifcl=True,
    )
    # generate movemap
    mm = gen_movemap(jump=True, chi=True, bb=True)

    for pose in poses:
        print_timestamp("Relaxing with ligand", start_time)
        pose.update_residue_neighbors()
        scores = dict(pose.scores)
        original_pose = pose.clone()
        # relax pose
        fast_relax(
            pose=pose,
            task_factory=task_factory,
            scorefxn=scorefxn,
            movemap=mm,
            relax_script=relax_script,
            repeats=1,
            cartesian=True,
            coord_constrain_sidechains=True,
            constrain_relax_to_start_coords=True,
        )
        print_timestamp("Relax complete", start_time)
        run_ligand_dock(pose, holo=scores["path_in"])
        print_timestamp("Ligand dock complete", start_time)
        scores.update(pose.scores)
        for key, value in scores.items():
            pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, key, value)
        ppose = io.to_packed(pose)
        yield ppose


def run_ligand_dock(pose: Pose, **kwargs) -> None:
    """
    :param: pose: Pose object to run GALigandDock on.
    :param: kwargs: Keywords to pass to GALigandDock.
    :return: None.
    """
    import pyrosetta

    if "holo" in kwargs:
        holo = kwargs["holo"]
    else:
        raise ValueError("Must specify holo keyword argument.")

    xml_snippet = f"""
    <SCOREFXNS>
        <ScoreFunction name="dockscore" weights="beta">
            <Reweight scoretype="fa_rep" weight="0.2"/>
        </ScoreFunction>
        <ScoreFunction name="relaxscore" weights="beta_cart">
            <Reweight scoretype="fa_rep" weight="0.2"/>
            <Reweight scoretype="coordinate_constraint" weight="0.1"/>
        </ScoreFunction>
    </SCOREFXNS>
    <MOVERS>
        <GALigandDock name="dock" scorefxn="dockscore" scorefxn_relax="relaxscore" 
        runmode="dockflex" nativepdb="{holo}" final_exact_minimize="bbsc1"
        cartmin_lig="1" padding="6.0" sidechains="aniso" >
            <Stage repeats="10" npool="50" pmut="0.2" smoothing="0.375" 
            rmsdthreshold="1.0" maxiter="50" pack_cycles="100" 
            ramp_schedule="0.1,1.0"/>
        </GALigandDock>
    </MOVERS>
    """
    objs = pyrosetta.rosetta.protocols.rosetta_scripts.XmlObjects.create_from_string(
        xml_snippet
    )
    docker = objs.get_mover("dock")
    docker.apply(pose)
    return


@requires_init
def check_linker_accessibility(
    packed_pose_in: Optional[PackedPose] = None, **kwargs
) -> Iterator[PackedPose]:
    """
    :param: packed_pose_in: PackedPose object to filter.
    :param: kwargs: Keyword arguments to pass to this function.
    :return: Iterator of PackedPose objects.
    Check the atomic burial of linker atoms, as well as cms and ddg of the small molecule.
    """

    import sys
    from pathlib import Path
    from time import time

    import pyrosetta
    import pyrosetta.distributed.io as io
    from pyrosetta.rosetta.core.select.residue_selector import (
        ChainSelector,
        NeighborhoodResidueSelector,
        OrResidueSelector,
    )

    # insert the root of the repo into the sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from crispy_shifty.protocols.cleaning import path_to_pose_or_ppose
    from crispy_shifty.protocols.design import (
        gen_scorefxn,
        gen_task_factory,
        pack_rotamers,
        score_cms,  
        score_ddg,
    )
    from crispy_shifty.utils.io import print_timestamp

    start_time = time()

    # generate poses or convert input packed pose into pose
    if packed_pose_in is not None:
        poses = [io.to_pose(packed_pose_in)]
        pdb_path = "none"
    else:
        pdb_path = kwargs["pdb_path"]
        poses = path_to_pose_or_ppose(
            path=pdb_path, cluster_scores=True, pack_result=False
        )

    print_timestamp("Setting up for repack", start_time)

    # make selectors
    chA = ChainSelector(1)
    chB = ChainSelector(2)
    around_chB = NeighborhoodResidueSelector(chB, 15)
    interface_or_chB = OrResidueSelector(around_chB, chB)
    # make stuff for repacker
    scorefxn = gen_scorefxn(
        cartesian=False, res_type_constraint=False, hbonds=False, weights="ref2015"
    )
    scorefxn.set_weight(pyrosetta.rosetta.core.scoring.fa_intra_rep, 0.55)
    task_factory = gen_task_factory(
        design_sel=None,
        pack_sel=interface_or_chB,
    )

    # check for atom_ids kwarg
    try:
        atom_ids = kwargs["atom_ids"]
    except KeyError:
        raise ValueError("Must specify atom_ids keyword argument.")
    # repack allowing ligand chi dofs
    for pose in poses:
        print_timestamp("Repacking ligand linker", start_time)
        pose.update_residue_neighbors()
        scores = dict(pose.scores)
        original_pose = pose.clone()
        # repack linker
        pack_rotamers(
            pose=pose,
            scorefxn=scorefxn,
            task_factory=task_factory,
        )
        # get linker burial
        linker_burial = read_lig_atm_sasa(pose, atom_ids.split(","), probe_radius=3.0)
        # rechain
        sc = pyrosetta.rosetta.protocols.simple_moves.SwitchChainOrderMover()
        sc.chain_order("12")
        sc.apply(pose)
        # score cms
        cms = score_cms(pose=pose, sel_1=chA, sel_2=chB)
        # score ddg
        ddg = score_ddg(pose=pose)
        # update scores
        scores.update(pose.scores)
        scores["linker_burial"] = linker_burial
        print_timestamp("Filtering complete", start_time)
        for key, value in scores.items():
            pyrosetta.rosetta.core.pose.setPoseExtraScore(pose, key, value)
        ppose = io.to_packed(pose)
        yield ppose


def read_lig_atm_sasa(pose: Pose, atmName_to_read=List[str], probe_radius=1.4) -> float:
    """
    :param: pose: Pose object to read sasa from.
    :param: atmName_to_read: List of atom names to read sasa for.
    :param: probe_radius: Radius of the probe sphere.
    :return: SASA of the ligand.
    GRL wrote this function.
    """

    import copy
    import numpy as np
    import pyrosetta

    atm_depth_s = pyrosetta.rosetta.core.scoring.atomic_depth.atomic_depth(
        pose, probe_radius
    )
    lig_resNo = pose.total_residue()
    lig_res = pose.residue(lig_resNo)
    res_natm = pose.pdb_info().natoms(lig_resNo)
    # this returns all atom values in list for the residue

    tmp_atm_names = []
    for i_atm_loc in range(1, res_natm + 1):
        atm_name = lig_res.atom_name(i_atm_loc)
        tmp_atm_names.append(atm_name)

    atmNames = []
    if len(atmName_to_read) > 0:
        for atmName in tmp_atm_names:
            if atmName.strip() in atmName_to_read:
                atmNames.append(atmName)
    else:
        atmNames = copy.deepcopy(tmp_atm_names)

    atom_depths = []
    for atm_name in atmNames:
        atm_id = pyrosetta.rosetta.core.id.AtomID(
            lig_res.atom_index("%s" % atm_name), lig_resNo
        )
        atom_depths.append(atm_depth_s(atm_id))
    return np.sum(atom_depths)
