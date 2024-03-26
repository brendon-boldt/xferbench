from pathlib import Path
from pprint import pprint
import re

import numpy as np
import pandas as pd
from scipy import stats  # type: ignore
from tqdm import tqdm  # type: ignore

from . import run
from .model import config

pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.max_rows", None)


def clm_analysis(rc: run.RunConfig) -> None:
    assert rc.save_prefix is not None
    base_path = Path(rc.save_prefix)
    save_path = get_analysis_path(base_path)

    df = get_clm_df(base_path, rc.overwrite)
    df = df.loc[df.index != "evtimova"]
    target_langs = df.columns.tolist()

    means = pd.DataFrame(df.mean(1))
    means = means.rename(columns={0: "mean"})

    normed = (df - df.mean()) / df.std()

    cis = pd.DataFrame()
    cis.index = normed.index
    cis["low"] = ""
    cis["high"] = ""
    cis["mean"] = normed.mean(1)
    for idx in df.index:
        arr = normed.loc[idx].to_numpy()
        v = stats.bootstrap([arr], np.mean)
        bs_mean = v.bootstrap_distribution.mean()
        ci = v.confidence_interval
        cis.loc[idx, "low"] = ci.low
        cis.loc[idx, "high"] = ci.high
        # means.loc[idx, "re_mean"] = (ci.high - bs_mean) * means.std() + means.mean()
        std = means["mean"].std()
        mean = means["mean"].mean()
        means.loc[idx, "minus"] = -(ci.low - cis.loc[idx, "mean"]) * std
        means.loc[idx, "plus"] = (ci.high - cis.loc[idx, "mean"]) * std

    means.index.name = "key"
    cols = ("name", "group", "xpos")
    means.loc[:, cols] = None
    df.loc[:, cols] = None

    for i, (k, v) in enumerate(LANG_METADATA.items()):
        means.loc[k, cols] = v + (i + 1,)  # type: ignore[call-overload]
        df.loc[k, cols] = v + (i + 1,)

    means = means.sort_values("xpos")

    df = df.sort_values("xpos")

    df.loc["mean"] = None
    df.loc[:, "mean"] = float("nan")
    df.loc["mean", "name"] = "\\textit{Mean}"
    df.loc["mean", "xpos"] = len(df)
    for tl in target_langs:
        df.loc["mean", tl] = df[tl].mean()
    df.loc[:, "mean"] = df.loc[:, target_langs].mean(1)

    print_tex_minmaxes(df)
    print_tex_minmaxes(means)

    # print(df.mean())
    print(df)
    # print(means)
    # print(cis)
    print(means)
    means.to_csv(save_path / "ce-means.tsv", sep="\t")
    df.to_csv(save_path / "clm.tsv", sep="\t")


def print_tex_minmaxes(df: pd.DataFrame) -> None:
    mk_tex_def = lambda mm, col, val: f"\\def\\{mm}{col}{{{val}}}"
    for col in df.columns:
        if not df[col].dtype == np.float64:
            continue
        print(mk_tex_def("min", col, df[col].min()), end=" ")
        print(mk_tex_def("max", col, df[col].max()), end=" ")
    print()


WRITING_SYSTEM_METADATA = {
    "fr": ("Latin", "Alphabet"),
    "es": ("Latin", "Alphabet"),
    "ru": ("Cyrillic", "Alphabet"),
    "zh": ("Chinese", "Logographic"),
    "ko": ("Hangul", "Alphabet"),
    "ar": ("Arabic", "Abjad"),
    "hi": ("Devanagari", "Abugida"),
    "da": ("Latin", "Alphabet"),
    "eu": ("Latin", "Alphabet"),
    "fa": ("Arabic", "Abjad"),
    "fi": ("Latin", "Alphabet"),
    "he": ("Hebrew", "Abjad"),
    "id": ("Latin", "Alphabet"),
    "ja": ("Japanese", "Mixed"),
    "kk": ("Cyrillic", "Alphabet"),
    "ro": ("Latin", "Alphabet"),
    "ur": ("Arabic", "Abjad"),
}


def clm_writing_system(rc: run.RunConfig) -> None:
    assert rc.save_prefix is not None
    base_path = Path(rc.save_prefix)
    save_path = get_analysis_path(base_path)

    df = get_clm_df(base_path, rc.overwrite)
    df = df.loc[df.index != "evtimova"]
    target_langs = df.columns.tolist()

    keep = lambda x: LANG_METADATA[x][1] == "human"
    df = df.loc[df.index.to_series().apply(keep)]
    normed = (df - df.mean()) / df.std()

    wsm = WRITING_SYSTEM_METADATA
    for metadata_index in 0, 1:
        grouper = [
            lambda x: wsm[x[0]][metadata_index],
            lambda x: wsm[x[1]][metadata_index],
        ]
        grouped = normed.stack().groupby(grouper).mean().unstack()
        grouped.index.name = "Source"
        for mm in "min", "max":
            val = getattr(getattr(grouped, mm)(), mm)()
            print(f"\\def\\{mm}val{{{val}}}", end=" ")
        print()
        print(grouped)
        print()
        name_postfix = ["", "-type"][metadata_index]
        grouped.to_csv(save_path / f"clm-writing-system{name_postfix}.tsv", sep="\t")


def get_analysis_path(path: Path) -> Path:
    new_path = path / "analysis"
    new_path.mkdir(parents=True, exist_ok=True)
    return new_path


def get_clm_df(
    path: Path, overwrite: bool, fail_no_cache: bool = False
) -> pd.DataFrame:
    save_path = get_analysis_path(path)
    fn_pickle = save_path / "clm.pkl"
    fn_csv = save_path / "clm.csv"

    if not overwrite:
        try:
            return pd.read_pickle(fn_pickle)
        except Exception as e:
            if fail_no_cache:
                raise e

    df = pd.DataFrame()
    for result_path in path.glob("*/*/result.txt"):
        source = result_path.parts[-3]
        target = result_path.parts[-2].split("-")[0]
        with result_path.open() as fo:
            df.loc[source, target] = float(fo.read())

    df.to_csv(fn_csv)
    df.to_pickle(fn_pickle)
    return df


LANG_METADATA = {
    "fr": ("French", "human"),
    "es": ("Spanish", "human"),
    "ru": ("Russian", "human"),
    "zh": ("Chinese", "human"),
    "ko": ("Korean", "human"),
    "ar": ("Arabic", "human"),
    "hi": ("Hindi", "human"),
    "pz_real": ("Paren, real", "synth"),
    "pz_syn": ("Paren, synth", "synth"),
    "disc_large": ("Disc, large", "ec"),
    "disc_small": ("Disc, small", "ec"),
    "recon_1": ("Rec, large", "ec"),
    "yao": ("Yao+", "ec"),
    "mu_sw": ("Mu+, SW", "ec"),
    "mu_cub": ("Mu+, CUB", "ec"),
    "rand": ("Random", "baseline"),
    "no_pt": ("No pretrain", "baseline"),
}


def correlation_analysis(rc: run.RunConfig) -> None:
    assert rc.save_prefix is not None
    base_path = Path(rc.save_prefix)
    save_path = get_analysis_path(base_path)

    mt_df = get_mt_df(base_path, False)
    clm_df = get_clm_df(base_path, False, fail_no_cache=True)
    clm_df = clm_df.loc[clm_df.index != "evtimova"]
    mt_df = mt_df.loc[mt_df.index != "evtimova"]
    clm_df.sort_index(inplace=True)
    mt_df.sort_index(inplace=True)

    clm_means = pd.DataFrame(clm_df.mean(1))
    clm_means = clm_means.rename(columns={0: "mean"})

    mt_means = mt_df.groupby(level=("source", "config")).mean()
    for mt_cfg in ["MtWmt", "MtFreeze", "MtLL"]:
        mask = mt_means.index.get_level_values(1) == mt_cfg
        filtered_mt = mt_means.loc[mask, "chrf"]
        print(mt_cfg)
        print(stats.kendalltau(clm_means["mean"], filtered_mt))
        print(stats.pearsonr(clm_means["mean"], filtered_mt))


def mt_analysis(rc: run.RunConfig) -> None:
    assert rc.save_prefix is not None
    base_path = Path(rc.save_prefix)
    save_path = get_analysis_path(base_path)

    df = get_mt_df(base_path, rc.overwrite)

    means = df.groupby(level=("source", "config")).mean()

    def process(which: str) -> pd.DataFrame:
        out = pd.DataFrame()
        i = 0
        for k, v in LANG_METADATA.items():
            if k not in means.index.levels[0]:  # type: ignore[attr-defined]
                continue
            out.loc[k, "name"] = v[0]
            out.loc[k, "xpos"] = i + 1
            out.loc[k, means.loc[k].index] = means.loc[k, which]
            i += 1

        out.index.name = "key"

        # means = means.sort_values("xpos")
        out["xpos"] = out["xpos"].astype(int)
        out.sort_values("xpos", inplace=True)

        out.to_csv(save_path / f"mt-{which}.tsv", sep="\t")
        print_tex_minmaxes(out)
        print(out)
        return out

    bleus = process("bleu")
    chrfs = process("chrf")


def get_mt_df(path: Path, overwrite: bool) -> pd.DataFrame:
    save_path = get_analysis_path(path)
    fn_pickle = save_path / "mt.pkl"
    fn_csv = save_path / "mt.csv"

    if not overwrite:
        try:
            return pd.read_pickle(fn_pickle)
        except:
            pass

    df = pd.DataFrame()
    df.index = pd.MultiIndex.from_tuples([], names=["source", "config", "bs"])
    result_paths = list(path.glob("*/*/result_*_bs*.txt"))
    assert len(result_paths) > 0, "No result files found."
    for result_path in tqdm(result_paths):
        source = result_path.parts[-3]
        cfg = re.search(r"(Mt[a-zA-Z]*)", result_path.parts[-2])[1]  # type: ignore[index]
        bs = re.search(r"bs([0-9]+)", result_path.parts[-1])[1]  # type: ignore[index]
        metric = re.search(r"result_([a-zA-Z]*)", result_path.parts[-1])[1]  # type: ignore[index]

        df.loc[(source, cfg, bs), metric] = -999
        with result_path.open() as fo:
            df.loc[(source, cfg, bs), metric] = float(fo.read())

    df.sort_index(inplace=True)
    df.to_csv(fn_csv)
    df.to_pickle(fn_pickle)
    return df
