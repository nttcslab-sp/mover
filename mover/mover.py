import collections
import decimal
import itertools
import numbers
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

import meeteval
from meeteval.io.ctm import CTM
from meeteval.io.seglst import SegLST
from meeteval.wer.preprocess import split_words
from mover._meeteval.wer.wer.time_constrained import align
from mover.get_speaker_mapping import get_speaker_mapping

NULL_TIME = 100000000
ALIGNMENT_EPS = "*"


def merge(alignment, num_ref=None, num_hyp=None):
    # Sanity check
    for s in alignment:
        ref, hyp = s
        if ref is not None:
            _num_ref = len(ref["words"]) if isinstance(ref["words"], list) else 1
            assert num_ref is None or num_ref == _num_ref, (num_ref, _num_ref)
            num_ref = _num_ref
        if hyp is not None:
            _num_hyp = len(hyp["words"]) if isinstance(hyp["words"], list) else 1
            assert num_hyp is None or num_hyp == _num_hyp, (num_hyp, _num_hyp)
            num_hyp = _num_hyp

    retval = []
    for s in alignment:
        segment = {}
        ref, hyp = s

        if isinstance(ref, dict) and hyp is None:
            segment = ref.copy()

            refwords = ref["words"] if isinstance(ref["words"], list) else [ref["words"]]
            ref_start_time = ref["start_time"] if isinstance(ref["start_time"], list) else [ref["start_time"]]
            ref_end_time = ref["end_time"] if isinstance(ref["end_time"], list) else [ref["end_time"]]
            org_ref_start_time = (
                ref["org_start_time"] if isinstance(ref["org_start_time"], list) else [ref["org_start_time"]]
            )
            org_ref_end_time = ref["org_end_time"] if isinstance(ref["org_end_time"], list) else [ref["org_end_time"]]

            segment["words"] = refwords + [ALIGNMENT_EPS] * num_hyp
            segment["start_time"] = ref_start_time + [NULL_TIME] * num_hyp
            segment["end_time"] = ref_end_time + [NULL_TIME] * num_hyp
            segment["org_start_time"] = org_ref_start_time + [NULL_TIME] * num_hyp
            segment["org_end_time"] = org_ref_end_time + [NULL_TIME] * num_hyp

            retval.append(segment)
        elif ref is None and isinstance(hyp, dict):
            segment = hyp.copy()

            hypwords = hyp["words"] if isinstance(hyp["words"], list) else [hyp["words"]]
            hyp_start_time = hyp["start_time"] if isinstance(hyp["start_time"], list) else [hyp["start_time"]]
            hyp_end_time = hyp["end_time"] if isinstance(hyp["end_time"], list) else [hyp["end_time"]]
            org_hyp_start_time = (
                hyp["org_start_time"] if isinstance(hyp["org_start_time"], list) else [hyp["org_start_time"]]
            )
            org_hyp_end_time = hyp["org_end_time"] if isinstance(hyp["org_end_time"], list) else [hyp["org_end_time"]]

            segment["words"] = [ALIGNMENT_EPS] * num_ref + hypwords
            segment["start_time"] = [NULL_TIME] * num_ref + hyp_start_time
            segment["end_time"] = [NULL_TIME] * num_ref + hyp_end_time
            segment["org_start_time"] = [NULL_TIME] * num_ref + org_hyp_start_time
            segment["org_end_time"] = [NULL_TIME] * num_ref + org_hyp_end_time

            retval.append(segment)

        elif isinstance(ref, dict) and isinstance(hyp, dict):
            segment = ref.copy()
            assert ref["speaker"] == hyp["speaker"], (ref["speaker"], hyp["speaker"])
            refwords = ref["words"] if isinstance(ref["words"], list) else [ref["words"]]
            ref_start_time = ref["start_time"] if isinstance(ref["start_time"], list) else [ref["start_time"]]
            ref_end_time = ref["end_time"] if isinstance(ref["end_time"], list) else [ref["end_time"]]
            org_ref_start_time = (
                ref["org_start_time"] if isinstance(ref["org_start_time"], list) else [ref["org_start_time"]]
            )
            org_ref_end_time = ref["org_end_time"] if isinstance(ref["org_end_time"], list) else [ref["org_end_time"]]

            hypwords = hyp["words"] if isinstance(hyp["words"], list) else [hyp["words"]]
            hyp_start_time = hyp["start_time"] if isinstance(hyp["start_time"], list) else [hyp["start_time"]]
            hyp_end_time = hyp["end_time"] if isinstance(hyp["end_time"], list) else [hyp["end_time"]]
            org_hyp_start_time = (
                hyp["org_start_time"] if isinstance(hyp["org_start_time"], list) else [hyp["org_start_time"]]
            )
            org_hyp_end_time = hyp["org_end_time"] if isinstance(hyp["org_end_time"], list) else [hyp["org_end_time"]]

            segment["words"] = refwords + hypwords
            segment["start_time"] = ref_start_time + hyp_start_time
            segment["end_time"] = ref_end_time + hyp_end_time
            segment["org_start_time"] = org_ref_start_time + org_hyp_start_time
            segment["org_end_time"] = org_ref_end_time + org_hyp_end_time

            retval.append(segment)
        elif ref is None and hyp is None:
            pass
        else:
            raise RuntimeError()

    return SegLST.new(retval)


def merge_like_doverlap(start_times, end_times, collar=10.0):
    xs = list(sorted(set(start_times + end_times)))
    if len(xs) == 1:
        return xs[0], xs[0]
    assert len(xs) > 0
    candidates = list(zip(xs[:-1], xs[1:]))

    voting = [0 for _ in xs[:-1]]
    for idx, (d1, d2) in enumerate(candidates):
        assert d1 < d2, "Bug"
        for st, et in zip(start_times, end_times):
            st, et = sorted([st, et])
            st -= collar
            et += collar
            if st <= d1 and d1 <= et:
                voting[idx] += 1
            # elif et <= d1 or d2 <= st:
            #     pass
            # else:
            #     raise RuntimeError((d1, d2, st, et))

    voted = [(d1, d2) for idx, (d1, d2) in enumerate(candidates) if 2 * voting[idx] >= len(start_times)]
    assert len(voted) > 0, (voting, candidates, len(start_times), xs)
    return voted[0][0], voted[-1][1]


def merge_center_points(start_times, end_times):
    return np.mean(start_times), np.mean(end_times)


def most_common_word(iterables):
    counter = collections.Counter(iterables)
    word, _ = counter.most_common()[0]
    return word


def vote(seglst, time_merge_type="center_points", collar=0.0) -> SegLST:
    if time_merge_type == "keep_segment":
        if len(seglst) > 0:
            # Get start_time from first token
            for index in range(len(seglst)):
                word = most_common_word(seglst[index]["words"])
                if word == ALIGNMENT_EPS:
                    continue
                start_times = [t for t, w in zip(seglst[index]["start_time"], seglst[index]["words"]) if w == word]
                break
            else:
                return SegLST.new([])

            for index in range(len(seglst) - 1, -1, -1):
                word = most_common_word(seglst[index]["words"])
                if word == ALIGNMENT_EPS:
                    continue
                end_times = [t for t, w in zip(seglst[index]["end_time"], seglst[index]["words"]) if w == word]
                break
            else:
                return SegLST.new([])

            start_time, end_time = merge_center_points(start_times, end_times)
            if end_time < start_time:
                # swap time
                start_time, end_time = end_time, start_time

            words = []
            for segment in seglst:
                if isinstance(segment["words"], list):
                    assert all(len(w.split()) == 1 for w in segment["words"])
                    assert all(len(w.split()) == 1 for w in segment["words"])
                    word = most_common_word(segment["words"])
                    if word != ALIGNMENT_EPS:
                        words.append(word)
                else:
                    words.append(segment["words"])
            words = " ".join(words)
            return SegLST.new(
                [
                    {
                        **seglst[0],
                        "words": words,
                        "start_time": start_time,
                        "end_time": end_time,
                    }
                ]
            )
        else:
            return SegLST.new([])

    else:
        retval = []
        for segment in seglst:
            out = segment.copy()
            if isinstance(segment["words"], list):
                assert all(len(w.split()) == 1 for w in segment["words"])
                word = most_common_word(segment["words"])
                out["words"] = word
                if word != ALIGNMENT_EPS:
                    start_times = [t for t, w in zip(segment["start_time"], segment["words"]) if w == word]
                    end_times = [t for t, w in zip(segment["end_time"], segment["words"]) if w == word]
                    if time_merge_type == "doverlap_like":
                        start_time, end_time = merge_like_doverlap(start_times, end_times, collar=collar)
                    elif time_merge_type == "center_points":
                        start_time, end_time = merge_center_points(start_times, end_times)
                    else:
                        raise RuntimeError(time_merge_type)
                    out["start_time"] = start_time
                    out["end_time"] = end_time
                    retval.append(out)
            else:
                retval.append(out)

        return SegLST.new(retval)


class Activity:
    def __init__(self):
        self.intervals = []

    def add(self, start_time, end_time):
        if start_time == end_time:
            return
        assert start_time < end_time, (start_time, end_time)
        for idx, (st, et) in enumerate(self.intervals):
            assert st < et
            if st > end_time:
                """
                            ------
                ---------
                """
                self.intervals.insert(idx, (start_time, end_time))
                break

            elif st <= start_time and end_time <= et:
                """
                ------
                ---

                 ------
                   --

                 ------
                 ------
                """
                break

            elif et < start_time:
                """
                ------
                       --
                """
                continue

            elif st >= start_time and et >= end_time >= st:
                """
                  ------
                -----

                  ------
                --

                  ------
                --------

                """
                self.intervals[idx] = (start_time, et)
                break

            elif start_time <= et and et <= end_time:
                """
                  ------
                    ---------

                  ------
                        -----

                  ------
                ------------
                """
                for idx2, (st2, et2) in enumerate(self.intervals[idx + 1 :], idx + 1):
                    if st2 > end_time:
                        del self.intervals[idx + 1 : idx2]
                        self.intervals[idx] = (min(st, start_time), end_time)
                        break
                    elif st2 <= end_time <= et2:
                        del self.intervals[idx + 1 : idx2 + 1]
                        self.intervals[idx] = (min(st, start_time), et2)
                        break
                    elif end_time > et2:
                        continue
                    else:
                        raise RuntimeError()
                else:
                    del self.intervals[idx + 1 :]
                    self.intervals[idx] = (min(st, start_time), end_time)
                    break
                break
            else:
                raise RuntimeError((start_time, end_time, st, et))
        else:
            self.intervals.append((start_time, end_time))
        for (st, et), (st2, et2) in zip(self.intervals, self.intervals[1:]):
            assert st < et, (st, et)
            assert st2 < et2, (st2, et2)
            assert et < st2, (et, st2)

    @classmethod
    def new_from_seglst(cls, seglst: SegLST):
        activity = Activity()
        for seg in seglst:
            activity.add(seg["start_time"], seg["end_time"])
        return activity

    def overlap_ratio(self, activity):
        total_length = sum(e - s for s, e in self.intervals)
        total_length2 = sum(e - s for s, e in activity.intervals)
        overlap_length = 0
        for interval1 in self.intervals:
            s1, e1 = interval1

            for interval2 in activity.intervals:
                s2, e2 = interval2
                overlap_length += max(min(e1, e2) - max(s1, s2), 0)
        if total_length > 0:
            ratio = overlap_length / total_length
        else:
            ratio = 0
        if total_length2 > 0:
            ratio2 = overlap_length / total_length2
        else:
            ratio2 = 0
        assert ratio <= 1.0
        return ratio, ratio2, overlap_length

    def thresholding(self, threshold, start_index=0):
        if threshold <= 0:
            return

        for idx, ((st, et), (st2, et2)) in enumerate(
            zip(self.intervals[start_index:], self.intervals[start_index + 1 :]),
            start_index,
        ):
            assert st < et, (st, et)
            assert st2 < et2, (st2, et2)
            assert et < st2, (et, st2)
            if st2 - et <= threshold:
                self.intervals[idx] = (st, et2)
                del self.intervals[idx + 1]
                break
        else:
            return
        self.thresholding(threshold, idx)


def split_by_long_pause(targets, threshold=0.0):
    eps = 0.000001
    threshold = max(threshold, eps)
    eps = get_type_of_start_time(targets)(eps)
    threshold = get_type_of_start_time(targets)(threshold)

    act = Activity()
    for target in targets:
        for seg in target:
            act.add(seg["start_time"], seg["end_time"])

    act.thresholding(threshold)

    splits = {(st, et): [[] for _ in targets] for st, et in act.intervals}
    for idx, target in enumerate(targets):
        for seg in target:
            if seg["start_time"] >= seg["end_time"]:
                continue
            for st, et in act.intervals:
                if (st - eps <= seg["start_time"] <= et + eps) and (st - eps <= seg["end_time"] <= et + eps):
                    splits[(st, et)][idx].append(seg)
                    break
            else:
                for st, et in act.intervals:
                    print(
                        st,
                        et,
                        (st - eps <= seg["start_time"] <= et + eps),
                        (st - eps <= seg["end_time"] <= et + eps),
                    )
                print(seg["start_time"] >= seg["end_time"])
                raise RuntimeError(seg)

    for key, value in splits.items():
        if all(len(v) == 0 for v in value):
            raise RuntimeError("")

    splits: Dict[Tuple[int, int], List[SegLST]] = {key: [SegLST.new(v) for v in value] for key, value in splits.items()}
    return splits


def apply_global_speaker_voting(targets: List[SegLST]) -> List[SegLST]:
    groupby_targets = [seg.groupby("session_id") for seg in targets]
    new_seglst_list = [[] for _ in targets]
    for key in groupby_targets[0]:
        _targets = [g[key] for g in groupby_targets]
        splits = split_by_long_pause(_targets, threshold=0)

        for split in splits.values():
            spk_counter = collections.Counter()

            num_spk = int(round(np.mean([len(s.groupby("speaker")) for s in split])))
            for seglst in split:
                for spk in seglst.groupby("speaker"):
                    spk_counter[spk] += 1

            spks = [t[0] for t in spk_counter.most_common(num_spk)]
            for seglst, new_seglst in zip(split, new_seglst_list):
                for seg in seglst:
                    if seg["speaker"] in spks:
                        new_seglst.append(seg)
    new_targets = [SegLST.new(seglst) for seglst in new_seglst_list]
    assert all(len(n.groupby("session_id")) == len(t.groupby("session_id")) for n, t in zip(new_targets, targets))
    return new_targets


def get_type_of_start_time(targets):
    for target in targets:
        for seg in target:
            if isinstance(seg["start_time"], decimal.Decimal):
                return decimal.Decimal
            else:
                return float
            break
        else:
            continue
        break
    return lambda x: x


def time_constrained_rover(
    targets: List[SegLST],
    tcdp_collar: Union[numbers.Real, decimal.Decimal] = 0,
    strategy="split",
    pseudo_word_level_timing="full_segment",
    time_merge_type="center_points",
    subset_grouping_threshold: numbers.Real = 0.0,
    empty_texts_for_absense_speaker: bool = True,
    speaker_voting: bool = False,
    overlap_ratio: numbers.Real = 0.0,
    order_consistency_resolution: bool = True,
    sctk_rover_path: Optional[Union[str, Path]] = None,
    sctk_rover_use_time: bool = False,
    sctk_rover_logdir: Union[str, Path] = None,
    sctk_rover_silence_duration: numbers.Real = 1.0,
) -> List[SegLST]:
    tcdp_collar = get_type_of_start_time(targets)(tcdp_collar)

    # Convert type of collar to correspond to tye type of seglst
    for iidx, target in enumerate(targets):
        for jidx, seg in enumerate(target):
            # Keep original start, end time
            seg["org_start_time"] = seg["start_time"]
            seg["org_end_time"] = seg["end_time"]

    groupby_targets = [seg.groupby(["session_id", "speaker"]) for seg in targets]

    # Derive all speakers for all sessions
    session_and_speakers = set(sum([list(dic.keys()) for dic in groupby_targets], []))

    merged: List[SegLST] = []
    for key in session_and_speakers:
        if empty_texts_for_absense_speaker:
            # Generate empty texts for absense speaker
            _targets = [dic[key] if key in dic else SegLST.new([]) for dic in groupby_targets]
        else:
            # else, remove the system from voting group
            _targets = [dic[key] for dic in groupby_targets if key in dic]

        ref = _targets[0]
        ref = ref.sorted("start_time")
        if strategy == "sctk":
            if sctk_rover_path is None:
                raise RuntimeError("Require sctk_rover_path")
            session_id, speaker = key
            out_seglst = sctk_rover(
                _targets,
                rover_path=sctk_rover_path,
                rover_logdir=Path(sctk_rover_logdir) / session_id / speaker,
                word_level_timing_strategy=pseudo_word_level_timing,
                silence_duration=sctk_rover_silence_duration,
                use_time=sctk_rover_use_time,
            )
            if len(out_seglst) > 0:
                merged.append(out_seglst)

        elif strategy in "subset":
            splits = split_by_long_pause(_targets, threshold=subset_grouping_threshold)

            for split in splits.values():
                ref = split[0]
                ref = ref.sorted("start_time")

                merged_flag = False
                num_ref = 0
                for hyp in split[1:]:
                    hyp = hyp.sorted("start_time")

                    num_ref += 1
                    alignment = align(
                        ref,
                        hyp,
                        style="seglst",
                        reference_pseudo_word_level_timing=pseudo_word_level_timing,
                        hypothesis_pseudo_word_level_timing=pseudo_word_level_timing,
                        collar=tcdp_collar,
                        reference_sort=False,
                        hypothesis_sort=False,
                        prune=False,
                        alignment_eps=ALIGNMENT_EPS,
                    )
                    ref = merge(alignment, num_ref=num_ref, num_hyp=1)
                    merged_flag = True

                if merged_flag:
                    _merged: SegLST = vote(ref, time_merge_type=time_merge_type, collar=tcdp_collar)
                else:
                    _merged = ref
                if len(_merged) > 0:
                    merged.append(_merged)

        elif strategy == "persegment":
            for idx, _ref in enumerate(ref):
                org = _ref
                _ref = SegLST.new([_ref])
                for idx2, hyp in enumerate(_targets[1:]):
                    assert len(ref) == len(hyp), (len(ref), len(hyp))
                    _hyp = hyp[idx]
                    assert _hyp["id"] == _ref[0]["id"], (_hyp["id"], _ref[0]["id"])
                    _hyp = SegLST.new([_hyp])
                    alignment = align(
                        _ref,
                        _hyp,
                        style="seglst",
                        reference_pseudo_word_level_timing="character_based_points",
                        hypothesis_pseudo_word_level_timing="character_based_points",
                        collar=tcdp_collar,
                        reference_sort=False,
                        hypothesis_sort=False,
                        prune=False,
                        alignment_eps=ALIGNMENT_EPS,
                    )
                    _ref = merge(alignment, num_ref=idx2 + 1, num_hyp=1)
                _merged = vote(_ref)
                words = " ".join(w["words"] for w in _merged)
                org["words"] = words
                org = SegLST.new([org])
                merged.append(org)

        elif strategy == "fullset":
            for idx2, hyp in enumerate(_targets[1:]):
                hyp = hyp.sorted("start_time")
                alignment = align(
                    ref,
                    hyp,
                    style="seglst",
                    reference_pseudo_word_level_timing=pseudo_word_level_timing,
                    hypothesis_pseudo_word_level_timing=pseudo_word_level_timing,
                    collar=tcdp_collar,
                    reference_sort=False,
                    hypothesis_sort=False,
                    prune=False,
                    alignment_eps=ALIGNMENT_EPS,
                )
                ref = merge(alignment, num_ref=idx2 + 1, num_hyp=1)
            _merged: SegLST = vote(ref, time_merge_type=time_merge_type, collar=tcdp_collar)
            if len(_merged) > 0:
                merged.append(_merged)

        else:
            raise RuntimeError(strategy)

    merged = SegLST.merge(*merged)
    assert len(merged.groupby("session_id")) == len(targets[0].groupby("session_id"))

    return merged


def overlap(segment1, segment2):
    if segment1["end_time"] >= segment2["start_time"]:
        return False
    elif segment2["end_time"] >= segment1["start_time"]:
        return False
    else:
        return True


def speaker_reassignment(reference, seglst_list, threshold=0.0):
    raise NotImplementedError("")

    assert isinstance(seglst_list, list), type(seglst_list)
    new_seglst_list = []
    one_or_more_speaker_changed_list = []
    for seglst in seglst_list:
        one_or_more_speaker_changed = False
        reference_group = reference.groupby("session_id")
        seglst_group = seglst.groupby("session_id")
        new_seglst = []
        for session_id in reference_group:
            reference_session = reference_group[session_id]
            seglst_session = seglst_group[session_id]
            splits: Dict[Tuple[float, float], List[SegLST]] = split_by_long_pause(
                [reference_session, seglst_session], threshold=threshold
            )
            for reference_split, seglst_split in splits.values():
                reference_split_group = reference_split.groupby("speaker")
                speaker_list = list(reference_split_group)
                seglst_split_group = seglst_split.groupby("speaker")
                base_error_dict = {
                    speaker: (
                        meeteval.wer.wer.time_constrained.time_constrained_siso_word_error_rate(
                            reference_split_group[speaker],
                            seglst_split_group[speaker],
                            collar=5,
                        ).error_rate
                        if speaker in seglst_split_group
                        else 1.0
                    )
                    for speaker in speaker_list
                }
                sorted_spearks = [
                    speaker
                    for speaker, error in sorted(
                        base_error_dict.items(),
                        key=lambda xs: xs[1] if xs[1] is not None else 1.0,
                    )
                ]
                if True:
                    org_indices = {}
                    for speaker in seglst_split_group:
                        for idx, seg in enumerate(seglst_split):
                            org_indices.setdefault(speaker, set()).add(idx)

                    # Overlap check
                    overlaps = []
                    for idx1, idx2 in itertools.combinations(range(len(seglst_split)), 2):
                        if overlap(seglst_split[idx1], seglst_split[idx2]):
                            overlaps.append((idx1, idx2))

                    non_overlap_segments_list = []
                    for num_seg in range(1, len(seglst_split) + 1):
                        for indices in itertools.combinations(range(len(seglst_split)), num_seg):
                            for idx1, idx2 in overlaps:
                                if not (idx1 in indices and idx2 in indices):
                                    non_overlap_segments_list.append(set(indices))

                    selected = set()
                    for speaker in sorted_spearks:
                        errors = []
                        for indices in non_overlap_segments_list:
                            if len(indices & selected) != 0:
                                continue

                            _target = SegLST.new([{**seglst_split[idx], "speaker": speaker} for idx in indices])
                            error = meeteval.wer.wer.time_constrained.time_constrained_siso_word_error_rate(
                                reference_split_group[speaker], _target, collar=5
                            ).error_rate
                            if error is None:
                                error = 1.0
                            errors.append((indices, _target, error))

                        if len(errors) > 0:
                            indices, _target, error = min(errors, key=lambda xs: xs[2])
                            if len(org_indices[speaker] & selected) > 0 or (
                                error < 0.5 and base_error_dict[speaker] - error > 0.05 * base_error_dict[speaker]
                            ):
                                selected = selected.union(indices)
                                new_seglst.append(_target)

                                # check if speaker was changed
                                assert any(seglst_split[idx]["speaker"] != speaker for idx in indices)
                                one_or_more_speaker_changed = True
                            else:

                                selected = selected.union(org_indices[speaker])
                                new_seglst.append(seglst_split_group[speaker])

                elif True:
                    seglst_split_split_group_list: List[SegLST] = [
                        v.groupby("speaker") for v, in split_by_long_pause([seglst_split], threshold=threshold).values()
                    ]
                    removed_speakers_list = [set() for _ in seglst_split_split_group_list]

                    for speaker in sorted_spearks:
                        pairs = list(
                            itertools.product(
                                *[
                                    set(seglst_split_split_group) - removed_speakers
                                    for seglst_split_split_group, removed_speakers in zip(
                                        seglst_split_split_group_list,
                                        removed_speakers_list,
                                    )
                                ]
                            )
                        )

                        errors = []
                        for pair in pairs:
                            assert len(pair) == len(seglst_split_split_group_list)
                            _target = SegLST.new(
                                [
                                    {**seg, "speaker": speaker}
                                    for speaker2, seglst_split_split_group in zip(pair, seglst_split_split_group_list)
                                    for seg in seglst_split_split_group[speaker2]
                                ]
                            )
                            error = meeteval.wer.wer.time_constrained.time_constrained_siso_word_error_rate(
                                reference_split_group[speaker], _target, collar=5
                            ).error_rate
                            if error is None:
                                error = 1.0
                            errors.append((pair, _target, error))

                        if len(errors) > 0:
                            pair, _target, error = min(errors, key=lambda xs: xs[2])

                            for speaker2, removed_speakers in zip(pair, removed_speakers_list):
                                removed_speakers.add(speaker2)

                            new_seglst.append(_target)

                            # check if speaker was changed
                            for speaker2 in pair:
                                if speaker2 != speaker:
                                    one_or_more_speaker_changed = True

                        else:
                            break

        new_seglst_list.append(SegLST.merge(*new_seglst))
        one_or_more_speaker_changed_list.append(one_or_more_speaker_changed)

    return new_seglst_list, one_or_more_speaker_changed_list


def apply_mapping(seglst_list, mapping):
    returns = []
    for idx, seglst in enumerate(seglst_list):
        new_seglst = []
        for _seg in seglst:
            _mapping, sorted_idx = mapping[_seg["session_id"]]
            _seg["speaker"] = _mapping[(idx, _seg["speaker"])]
            new_seglst.append(_seg)
        returns.append(SegLST.new(new_seglst))
    return returns


def seglst2ctm(
    infile,
    word_level_timing_strategy="character_based",
    segment_representation="word",
):
    if isinstance(infile, (SegLST, list)):
        seglst = infile
    else:
        seglst = meeteval.io.load(infile)

    if len(infile) == 0:
        infinity = 100000000000000000000
        lis = [
            dict(
                session_id="dummy",
                start_time=infinity,
                end_time=infinity,
                words="@",
                speaker="dummy",
                channel=0,
                confidence=0.1,
            )
        ]

    else:
        seglst = seglst.sorted("start_time")
        seglst = split_words(
            seglst,
            word_level_timing_strategy=word_level_timing_strategy,
            segment_representation=segment_representation,
        )

        lis = []
        for seg in seglst:
            # Assume per-session and per-speaker
            lis.append(
                dict(
                    session_id="dummy",
                    start_time=seg["start_time"],
                    end_time=seg["end_time"],
                    words=seg["words"],
                    speaker="dummy",
                    channel=0,
                    confidence=0.1,
                )
            )

    seglst = SegLST.new(lis)
    ctm = CTM.new(seglst)
    return ctm


def sctk_rover(
    inseglst_list: List[SegLST],
    rover_path: Union[str, Path],
    word_level_timing_strategy="character_based",
    silence_duration: numbers.Real = 1.0,
    rover_logdir=None,
    use_time: bool = False,
) -> List[SegLST]:
    if not Path(rover_path).exists():
        raise RuntimeError(f"No such file or directory: {rover_path}")

    for seglst in inseglst_list:
        if len(seglst) > 0:
            session_id = seglst[0]["session_id"]
            speaker = seglst[0]["speaker"]
            break
    else:
        raise RuntimeError(inseglst_list)

    out_seglst = []
    if rover_logdir is None:
        rover_logdir = tempfile.TemporaryDirectory()
        outdir = Path(rover_logdir.name)
    else:
        outdir = Path(rover_logdir)
    result_file = outdir / "merged.ctm"

    # Convert seglst to ctm
    outfiles = []
    for i, _seglst in enumerate(inseglst_list):
        ctm = seglst2ctm(_seglst, word_level_timing_strategy=word_level_timing_strategy)
        outfile = outdir / f"system{i}.ctm"
        outfile.parent.mkdir(parents=True, exist_ok=True)
        ctm.dump(outfile)
        outfiles.append(outfile)

    # Run rover
    cargs = [str(rover_path)]
    for outfile in outfiles:
        cargs += ["-h", str(outfile), "ctm"]
    cargs += [
        "-o",
        str(result_file),
        "-m",
        "meth1",
    ]
    if silence_duration != 1.0:
        raise RuntimeError("Not supporting: silence_duration!=1.0")

        cargs += ["-d", str(silence_duration)]

    if use_time:
        cargs += ["-T"]

    subprocess.run(cargs, check=True)

    # To sync
    number_of_retryings = 1000
    for _ in range(number_of_retryings):
        list(Path(result_file).parent.glob("*"))
        if result_file.exists():
            break
    else:
        raise RuntimeError(result_file)

    # Merge CTM into a seglst file
    ctmgroup = meeteval.io.load(result_file)
    for _, ctm in ctmgroup.ctms.items():
        for line in ctm:
            if line.word != "@":
                out_seglst.append(
                    dict(
                        session_id=session_id,
                        start_time=line.begin_time,
                        end_time=(line.begin_time + line.duration),
                        words=line.word,
                        speaker=speaker,
                        confidence=getattr(line, "confidence"),
                    )
                )
    return SegLST.new(out_seglst)


def apply_order_consistency_resolution(seglst):
    new_seglst = []
    for _seglst in seglst.groupby(["session_id", "speaker"]).values():
        while True:
            for idx, (seg, next_seg) in enumerate(zip(_seglst[:-1], _seglst[1:])):
                if seg["end_time"] > next_seg["start_time"]:
                    start_time = min(seg["start_time"], next_seg["start_time"])
                    end_time = max(seg["end_time"], next_seg["end_time"])
                    words = seg["words"] + " " + next_seg["words"]
                    new_seg = {
                        **seg,
                        "start_time": start_time,
                        "end_time": end_time,
                        "words": words,
                    }
                    _seglst = _seglst[:idx] + [new_seg] + _seglst[idx + 2 :]
                    break
            else:
                break
        new_seglst += _seglst
    return SegLST.new(new_seglst)


def mover(
    seglst_list: List[Union[str, Path, SegLST]],
    tcdp_collar: Union[numbers.Real, decimal.Decimal] = 0,
    strategy: str = "fullset",
    pseudo_word_level_timing: str = "character_based",
    time_merge_type: str = "center_points",
    subset_grouping_threshold: numbers.Real = 0.0,
    empty_texts_for_absense_speaker: bool = False,
    speaker_mapping: bool = True,
    speaker_voting: bool = False,
    overlap_ratio: numbers.Real = 0.0,
    order_consistency_resolution: bool = True,
    sctk_rover_path: Optional[Union[str, Path]] = None,
    sctk_rover_logdir: Optional[Union[str, Path]] = None,
    sctk_rover_use_time: bool = False,
    sctk_rover_silence_duration: numbers.Real = 1.0,
) -> List[SegLST]:
    seglst_list = [meeteval.io.load(t) if isinstance(t, (str, Path)) else t for t in seglst_list]
    if len(seglst_list) == 0:
        raise RuntimeError("len(seglst_list) == 0")
    elif len(seglst_list) == 1:
        warnings.warn("len(seglst_list) == 1")
        return seglst_list[0]

    if speaker_mapping:
        mapping = get_speaker_mapping(seglst_list)
        seglst_list = apply_mapping(seglst_list, mapping)

    if speaker_voting:
        seglst_list = apply_global_speaker_voting(seglst_list)

    merged = time_constrained_rover(
        targets=seglst_list,
        tcdp_collar=tcdp_collar,
        strategy=strategy,
        pseudo_word_level_timing=pseudo_word_level_timing,
        time_merge_type=time_merge_type,
        subset_grouping_threshold=subset_grouping_threshold,
        empty_texts_for_absense_speaker=empty_texts_for_absense_speaker,
        overlap_ratio=overlap_ratio,
        sctk_rover_path=sctk_rover_path,
        sctk_rover_logdir=sctk_rover_logdir,
        sctk_rover_use_time=sctk_rover_use_time,
        sctk_rover_silence_duration=sctk_rover_silence_duration,
    )

    if order_consistency_resolution:
        merged = apply_order_consistency_resolution(merged)
    return merged
