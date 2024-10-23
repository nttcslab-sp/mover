import argparse
import glob
from pathlib import Path

import meeteval
from mover.mover import mover


def str2bool(arg):
    if isinstance(arg, bool):
        return arg
    elif arg.lower().strip() in ["true", "1"]:
        return True
    elif arg.lower().strip() in ["false", "0"]:
        return
    else:
        raise ValueError(arg)


def cli(cmd=None):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--infile", type=str, required=True, nargs="+")
    parser.add_argument("--outfile", type=str, required=True)
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["subset", "fullset", "sctk", "persegment"],
        default="fullset",
    )
    parser.add_argument(
        "--time-merge-type",
        type=lambda x: str(x).replace("-", "_"),
        choices=["center-points", "keep-segment"],
        default="center_points",
    )
    parser.add_argument(
        "--pseudo-word-level-timing",
        type=lambda x: str(x).replace("-", "_"),
        choices=[
            "character-based",
            "character-based-points",
            "full-segment",
            "equidistant-intervals",
            "equidistant-points",
        ],
        default="character_based",
    )
    parser.add_argument("--subset-grouping-threshold", type=float, default=0.0)
    parser.add_argument("--tcdp-collar", type=float, default=5.0)
    parser.add_argument("--speaker-mapping", type=str2bool, default=True)
    parser.add_argument("--speaker-voting", type=str2bool, default=False)
    parser.add_argument("--empty-texts-for-absense-speaker", type=str2bool, default=False)
    parser.add_argument("--overlap-ratio", type=float, default=0.0)
    parser.add_argument("--order_consistency_resolution", type=str2bool, default=True)
    parser.add_argument("--sctk-rover-logdir", type=str, default=None)
    parser.add_argument("--sctk-rover-path", type=str, default=None)
    parser.add_argument("--sctk-rover-use-time", type=str2bool, default=False)
    parser.add_argument("--sctk-rover-silence-duration", type=float, default=1.0)

    args = parser.parse_args(cmd)

    infiles = []
    for infile in args.infile:
        matched = glob.glob(infile, recursive=True)
        if len(matched) == 0:
            raise RuntimeError(f"No such file or directory: {infile}")
        infiles += matched

    merged = mover(
        seglst_list=infiles,
        tcdp_collar=args.tcdp_collar,
        subset_grouping_threshold=args.subset_grouping_threshold,
        strategy=args.strategy,
        pseudo_word_level_timing=args.pseudo_word_level_timing,
        time_merge_type=args.time_merge_type,
        speaker_mapping=args.speaker_mapping,
        speaker_voting=args.speaker_voting,
        empty_texts_for_absense_speaker=args.empty_texts_for_absense_speaker,
        sctk_rover_logdir=args.sctk_rover_logdir,
        sctk_rover_path=args.sctk_rover_path,
        sctk_rover_use_time=args.sctk_rover_use_time,
        sctk_rover_silence_duration=args.sctk_rover_silence_duration,
        overlap_ratio=args.overlap_ratio,
        order_consistency_resolution=args.order_consistency_resolution,
    )

    outfile = Path(args.outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    merged.dump(outfile)


if __name__ == "__main__":
    cli()
