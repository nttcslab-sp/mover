<h1 align="center">MOVER</h1>
<h3 align="center">Combining Multiple Meeting Recognition Systems</h3>


This is the implementation of ["MOVER: Combining Multiple Meeting Recognition Systems" (Interspeech 2025)](https://www.isca-archive.org/interspeech_2025/kamo25_interspeech.html).

## Install


```
pip install git+https://github.com/nttcslab-sp/mover
```


## Command line tool


```sh
mover --infile ./example_files/sys1.json ./example_files/sys2.json ./example_files/sys3.json --outfile out.json

# Enable to handle wild-cards
mover --infile './example_files/*.json' --outfile out.json
```


**Warning! We will continue to refactor the API and update the documentation in the future. The names and usage of function arguments are subject to change.**

## Python API

```python
from mover import mover

seglst_object = mover(["./example_files/sys1.json", "./example_files/sys2.json", "./example_files/sys3.json"])
seglst_object.dump("out.json")
```

The JSON file format is "SEGment-wise Long-form Speech Transcription annotation (SegLST, see also [MeetEval](https://github.com/fgnt/meeteval/) about the format)", the file format used in [the CHiME challenges](http://chimechallenge.org/challenges/chime7/task1/index), and inside the function it is handled as a ``meeteval.io.SegLST`` instance via ``meeteval.io.load``. The return type of the ``mover`` function is also ``SegLST``.


Alternatively, ``SegLST`` instances can be passed directly as arguments.

```python
from mover import mover
import meeteval

seglst_list = [meeteval.io.load(f) for f in ["./example_files/sys1.json", "./example_files/sys2.json", "./example_files/sys3.json"]]

seglst_object = mover(seglst_list)
seglst_object.dump("out.json")
```

### TIPS

Since the output text from different recognition systems may differ in the notation style of numbers, symbols, and punctuations, it is recommended to perform normalization into your desired style before applying mover. For example, when applying chime_utils.text_norm.get_txt_norm, you can do as follows:

```sh
pip install git+https://github.com/chimechallenge/chime-utils
```

```python
from chime_utils.text_norm import get_txt_norm
text_norm_fn = get_txt_norm("chime8")

for seglst in seglst_list:
    for segment in seglst:
        words = segment["words"]
        for _ in range(5):
            words_ = text_norm_fn(words)
            if words == words_:
                break
            words = words_
        else:
            raise RuntimeError()
        segment["words"] = words
seglst_object = mover(seglst_list)
```


## Cite


[![ISCA DOI](https://img.shields.io/badge/ISCA/DOI-10.21437/Interspeech.2025--1614-blue.svg)](https://doi.org/10.21437/Interspeech.2025-1614)
[![arXiv](https://img.shields.io/badge/arXiv-2508.05055-b31b1b.svg)](https://arxiv.org/abs/2508.05055)



```
@inproceedings{kamo25_interspeech,
  title     = {{MOVER: Combining Multiple Meeting Recognition Systems}},
  author    = {{Naoyuki Kamo and Tsubasa Ochiai and Marc Delcroix and Tomohiro Nakatani}},
  year      = {{2025}},
  booktitle = {{Interspeech 2025}},
  pages     = {{3424--3428}},
  doi       = {{10.21437/Interspeech.2025-1614}},
}
```

