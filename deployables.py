# Python standard library
from typing import Iterator, Optional

# 3rd party library imports
# Rosetta library imports
from pyrosetta.distributed.packed_pose.core import PackedPose
from pyrosetta.distributed import requires_init

# Custom library imports


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

    from pathlib import Path
    import sys
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
        gen_scorefxn, gen_std_layer_design, gen_task_factory, interface_between_selectors, pack_rotamers
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

    from itertools import product
    from pathlib import Path
    import sys
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
    from crispy_shifty.protocols.mpnn import MPNNLigandDesign, MPNNDesign
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
        non_interface_indices = [str(i) for i, in_sel in enumerate(chA_not_interface.apply(pose), start=1) if in_sel]
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
        sc.apply(pose)
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
            pyrosetta.rosetta.core.pose.setPoseExtraScore(designed_pose, "type", design_method)
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

    from operator import lt, gt
    from pathlib import Path
    import sys
    from time import time
    import pyrosetta
    import pyrosetta.distributed.io as io
    from pyrosetta.rosetta.core.pose import Pose

    # insert the root of the repo into the sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from crispy_shifty.protocols.cleaning import path_to_pose_or_ppose
    from crispy_shifty.protocols.folding import (
        generate_decoys_from_pose,
        SuperfoldRunner,
    )
    from crispy_shifty.protocols.mpnn import dict_to_fasta, fasta_to_dict
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
        print_timestamp("Setting up for AF2", start_time)
        runner = SuperfoldRunner(
            pose=pose, fasta_path=fasta_path, load_decoys=True, **kwargs
        )
        runner.setup_runner(file=fasta_path)
        # initial_guess, reference_pdb both are the tmp.pdb
        initial_guess = str(Path(runner.get_tmpdir()) / "tmp.pdb")
        reference_pdb = initial_guess
        flag_update = {
            "--initial_guess": initial_guess,
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
            "mean_plddt": (gt, 90.0),
        }
        rank_on = "mean_plddt"
        prefix = "mpnn_seq"
        print_timestamp("Generating decoys", start_time)
        for decoy in generate_decoys_from_pose(
            pose,
            filter_dict=filter_dict,
            generate_prediction_decoys=True,
            label_first=True,
            prefix=prefix,
            rank_on=rank_on,
        ):
            packed_decoy = io.to_packed(decoy)
            yield packed_decoy


@requires_init
def fold_unbound(
    packed_pose_in: Optional[PackedPose] = None, **kwargs
) -> Iterator[PackedPose]:
    """
    :param: packed_pose_in: a PackedPose object to fold with the Superfold script.
    :param: kwargs: keyword arguments to be passed to the Superfold script.
    :return: an iterator of PackedPose objects.
    """

    import os
    import sys
    from pathlib import Path
    from time import time

    import pyrosetta
    import pyrosetta.distributed.io as io
    from pyrosetta.rosetta.core.pose import Pose

    # insert the root of the repo into the sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from crispy_shifty.protocols.cleaning import path_to_pose_or_ppose
    from crispy_shifty.protocols.mpnn import dict_to_fasta, fasta_to_dict
    from crispy_shifty.utils.io import cmd, print_timestamp

    start_time = time()
    # get the pdb_path from the kwargs
    pdb_path = kwargs.pop("pdb_path")
    # there are multiple paths in the pdb_path, we need to split them and rejoin them
    pdb_paths = pdb_path.split("____")
    pdb_path = " ".join(pdb_paths)

    # this function is special, we don't want a packed_pose_in ever, we maintain it as
    # a kwarg for backward compatibility with PyRosettaCluster
    if packed_pose_in is not None:
        raise ValueError("This function is not intended to have a packed_pose_in")
    else:
        pass

    print_timestamp("Setting up for AF2", start_time)
    runner = SuperfoldMultiPDB(input_file=pdb_path, load_decoys=True, **kwargs)
    runner.setup_runner(chains_to_keep=[1])
    print_timestamp("Running AF2", start_time)
    runner.apply()
    print_timestamp("AF2 complete, updating pose datacache", start_time)
    # get the updated poses from the runner
    tag_pose_dict = runner.get_tag_pose_dict()
    # filter the decoys
    filter_dict = {
        "mean_plddt": (gt, 90.0),
        "rmsd_to_input": (lt, 2.0),
    }
    rank_on = "mean_plddt"
    print_timestamp("Generating decoys", start_time)
    for tag, pose in tag_pose_dict.items():
        for decoy in generate_decoys_from_pose(
            pose,
            filter_dict=filter_dict,
            generate_prediction_decoys=False,
            label_first=False,
            prefix=tag,
            rank_on=rank_on,
        ):
            scores = dict(decoy.scores)
            bound_pose = None
            for original_path in pdb_paths:
                if tag in original_path:
                    bound_pose = next(
                        path_to_pose_or_ppose(
                            path=original_path, cluster_scores=True, pack_result=False
                        )
                    )
                    final_pose = Pose()
                    break
                else:
                    continue
            if bound_pose is None:
                raise RuntimeError
            else:
                pass
            for key, value in scores.items():
                pyrosetta.rosetta.core.pose.setPoseExtraScore(bound_pose, key, value)
            final_ppose = io.to_packed(final_pose)
            yield final_ppose

