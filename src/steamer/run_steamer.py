# This module was primarily developed by and taken from the Welch lab with some modifications
import csv
import gzip
import os
import time
from collections import defaultdict

from pathlib import Path, PurePath
import typer
from typing_extensions import Annotated

from ALLCools.mcds import MCDS
import warnings
import pandas as pd
import sparse
from fuc import pybed
from pybedtools import BedTool, Interval
from scipy.io import mmwrite
from scipy.sparse import coo_matrix, csr_matrix

app = typer.Typer()

@app.command()
def create_bed_for_TEs(filename: Path):
    """
    Takes in a file (ideally a file containing TEs) and converts it into a BED file.

    Args:
        filename: Path, filepath to the input file (ideally .rph.hits)

    Returns:
        bf: pybed.BedFrame object (and creating the TE bed file)
    """

    # col_names for the .rph.hits file
    col_names = ["seq_name", "ali-st", "ali-en"]
    other_col = ["family_name", "strand"]
    # only taking the columns with the col_names specified
    filename_str = filename.as_posix()
    if ".gz" in filename_str:
        TE_df = pd.read_csv(
            filename,
            sep="\t",
            usecols=[0, 9, 10],
            names=col_names,
            header=0,
            compression="gzip",
        )
    else:
        TE_df = pd.read_csv(
            filename, sep="\t", usecols=[0, 9, 10], names=col_names, header=0
        )
    TE_df.drop(index=TE_df.index[-1], axis=0, inplace=True)
    # strand and family name are in between chrm, start, and end so they have to be added at the end
    strand_name_df = pd.read_csv(
        filename, sep="\t", usecols=[2, 8], names=other_col, header=0
    )
    strand_name_df.drop(index=strand_name_df.index[-1], axis=0, inplace=True)
    # next we need to sort the strands out or else it will cause issues: the start strand cannot be greaer than the end
    TE_df["ali-st"] = TE_df["ali-st"].astype(int)
    TE_df["ali-en"] = TE_df["ali-en"].astype(int)
    for row in TE_df.itertuples():
        row_index, seq_name, start, end = row
        if start > end:
            TE_df.at[row_index, "ali-st"] = end
            TE_df.at[row_index, "ali-en"] = start
    # renaming columns to prepare for pybed.BedFrame.from_frame
    TE_df.rename(
        columns={"seq_name": "Chromosome", "ali-st": "Start", "ali-en": "End"},
        inplace=True,
    )
    TE_df = pd.concat([TE_df, strand_name_df], axis=1)
    TE_bf = pybed.BedFrame.from_frame(meta=[], data=TE_df)
    TE_bf_sort = TE_bf.sort()
    TE_bf_sort.to_file("TEs.bed")
    return TE_bf_sort

@app.command()
def create_bed_for_fragments(filename: Path, quality_barcode_file: Annotated[Path, typer.Argument()]):
    """
    Takes in a file (ideally a fragment file) and converts it into a BED file.

    Args:
        filename: str, filepath to the input file (ideally fragment file: .tsv)
        quality_barcode_file: str, filepath to the input file(file containing only the barcodes that passed QC)

    Returns:
        bf: pybed.BedFrame object (and creating the Fragment bed file)
    """

    col_names = ["Chromosome", "Start", "End", "barcode"]
    filename_str = filename.as_posix()
    if ".gz" in filename_str:
        frag_df = pd.read_csv(
            filename,
            sep="\t",
            usecols=[0, 1, 2, 3],
            names=col_names,
            comment="#",
            compression="gzip",
        )
    else:
        frag_df = pd.read_csv(
            filename, sep="\t", usecols=[0, 1, 2, 3], names=col_names, comment="#"
        )
    # list of valid mouse chromsomes
    valid_chromosomes = ["chr" + str(i) for i in range(1, 19)] + ["chrX", "chrY"]
    # drop any non valid chromosomes
    frag_df = frag_df[frag_df["Chromosome"].isin(valid_chromosomes)]
    if quality_barcode_file:
        if quality_barcode_file.exists() is False:
            frag_bf = pybed.BedFrame.from_frame(meta=[], data=frag_df)
            frag_bf_sort = frag_bf.sort()
            frag_bf_sort.to_file("Frag.bed")
        else:
            quality_barcodes_df = pd.read_csv(
                quality_barcode_file, sep="\t", names=["barcode"]
            )
            quality_barcodes = quality_barcodes_df["barcode"].tolist()
            frag_df = frag_df[frag_df["barcode"].isin(quality_barcodes)]
            frag_bf = pybed.BedFrame.from_frame(meta=[], data=frag_df)
            frag_bf_sort = frag_bf.sort()
            frag_bf_sort.to_file("Frag.bed")
    else:
        frag_bf = pybed.BedFrame.from_frame(meta=[], data=frag_df)
        frag_bf_sort = frag_bf.sort()
        frag_bf_sort.to_file("Frag.bed")

    return frag_bf_sort

def mangle_ids(line: Interval) -> Interval:
        fam_name = line.fields[3]
        TE_start, TE_end, chrom = (
            line.fields[1],
            line.fields[2],
            line.fields[0],
        )
        TE_name_unique = f"{fam_name}({chrom}:{TE_start},{TE_end})"
        line.name = TE_name_unique
        return line

@app.command()
def mangle_bed_file_ids(bed: Path, out: Path):
    loaded_bed = BedTool(bed)
    mangled_bed = loaded_bed.each(mangle_ids)
    mangled_bed.saveas(out)




def make_cell_x_element_matrix(bed_intersect, cell_barcodes=None):
    # Initialize lists and dictionaries
    if cell_barcodes is None:
        cell_barcodes = []
    row_indices = []  # Row indices for sparse matrix
    col_indices = []  # Column indices for sparse matrix (corresponding to UniqueTE)
    fams_col_indices = []  # Column indices for sparse matrix (corresponding to TE_Fam)
    data = []  # Data values for sparse matrix
    unique_TEs_list = defaultdict(
        lambda: len(unique_TEs_list)
    )
    TEs_fam_dict = defaultdict(
        lambda: len(TEs_fam_dict)
    )  # Dictionary to map TE_Fam names to indices
    barcode_dict = {
        barcode: i for i, barcode in enumerate(cell_barcodes)
    }  # Dictionary to map barcodes to indices

    # Iterate over bed_intersect
    for interval in bed_intersect:
        fam_name = interval.fields[3]
        barcode = interval.fields[8]
        TE_start, TE_end, chrom = (
            interval.fields[1],
            interval.fields[2],
            interval.fields[0],
        )
        TE_name_unique = f"{fam_name}({chrom}:{TE_start},{TE_end})"
        # Check if UniqueTE is already in unique_TEs_list, otherwise assign a new index
        UniqueTE_index = unique_TEs_list[TE_name_unique]
        # Check if TE_Fam is already in TEs_fam_dict, otherwise assign a new index
        TEFam_index = TEs_fam_dict[fam_name]
        # Get the barcode index from barcode_dict
        barcode_index = barcode_dict[barcode]
        # Append values to the respective lists
        row_indices.append(barcode_index)
        col_indices.append(UniqueTE_index)
        fams_col_indices.append(TEFam_index)
        data.append(1)
    # Create sparse matrices using coo_matrix
    UniqueTEs_sparse_matrix = coo_matrix((data, (row_indices, col_indices)))
    TE_Fams_sparse_matrix = coo_matrix((data, (row_indices, fams_col_indices)))
    # Convert sparse matrices to DataFrames and perform grouping
    UniqueTEs_sparse_df = pd.DataFrame(
        {
            "barcode_index": UniqueTEs_sparse_matrix.row,
            "UniqueTE_index": UniqueTEs_sparse_matrix.col,
            "data": UniqueTEs_sparse_matrix.data,
        }
    )
    UniqueTEs_sparse_df = UniqueTEs_sparse_df.groupby(
        ["barcode_index", "UniqueTE_index"], as_index=False
    )["data"].sum()
    TE_Fams_sparse_df = pd.DataFrame(
        {
            "barcode_index": TE_Fams_sparse_matrix.row,
            "FamTE_index": TE_Fams_sparse_matrix.col,
            "data": TE_Fams_sparse_matrix.data,
        }
    )
    TE_Fams_sparse_df = TE_Fams_sparse_df.groupby(
        ["barcode_index", "FamTE_index"], as_index=False
    )["data"].sum()
    # Return the DataFrames and dictionaries
    return (
        UniqueTEs_sparse_df,
        TE_Fams_sparse_df,
        dict(unique_TEs_list),
        dict(TEs_fam_dict),
        barcode_dict,
    )


def convert_cell_x_element_matrix_to_file(
    matrix, col, rows, filename="Cell_x_Element_Matrix.csv"
):
    """
    Converts a scipy sparse matrix into a CSV file.

    Args:
        matrix (csr_matrix): The input matrix to convert.
        col (list): A list of column labels for the output CSV file.
        rows (list): A list of row labels for the output CSV file.

    Returns:
        None
    """
    df = pd.DataFrame.sparse.from_spmatrix(matrix, columns=col, index=rows)
    df.to_csv(filename)


def save_df_as_gz(df, filename):
    # Save the DataFrame as a gzipped file
    with gzip.open(filename, "wt", encoding="utf-8") as gz_file:
        df.to_csv(gz_file, sep="\t", index=False, header=False)


def prepare_df(df):
    "Swaps the barcodes and TE column indexes to prepare for scanpy."
    df[df.columns[0]], df[df.columns[1]] = df[df.columns[1]], df[df.columns[0]]
    return df


def convert_df_to_sparse(df):
    """"""
    df = prepare_df(df)
    array = df.values
    rows = [val[0] for val in array]
    columns = [val[1] for val in array]
    values = [val[2] for val in array]
    csr_mat = csr_matrix((values, (rows, columns)))
    return csr_mat


def compress_tsv_file(file_name, output_dir, barcode):
    """
    Takes in a string,out_dir, and the barcodes to make a .gzip .tsv file
    """
    # Create the output file path
    output_file_path = PurePath.joinpath(Path(output_dir), file_name)
    # Write the TSV.GZ file
    with gzip.open(output_file_path, "wt", newline="") as tsv_file:
        writer = csv.writer(tsv_file, delimiter="\t")
        for barcode in barcode.keys():
            writer.writerow([barcode])

def make_features_df(TE_dict):
    # Convert TEs_fams dictionary to a DataFrame
    fams_df = pd.DataFrame(list(TE_dict.items()), columns=["TE_Name", "idx"])
    # Drop the 'idx' column
    fams_df.drop("idx", axis=1, inplace=True)
    # Move the 'pseudoID' column to the first position
    fams_df.insert(0, "pseudoID", fams_df["TE_Name"])
    # Add the 'expression' column
    fams_df["expression"] = ["Gene Expression"] * len(TE_dict)
    return fams_df


def compress_sparse_matrix(matrix, file_path):
    # Save the sparse matrix in Matrix Market format to a temporary uncompressed file
    mmwrite(file_path, matrix)
    # Compress the file using gzip
    with open(file_path, "rb") as f_in:
        with gzip.open(file_path + ".gz", "wb") as f_out:
            f_out.writelines(f_in)

    # Remove the temporary uncompressed file
    os.remove(file_path)


def display_elapsed_time(start_time, p=True):
    # Calculate the elapsed time
    elapsed_time_seconds = time.time() - start_time

    # Convert elapsed time to hours, minutes, and seconds
    hours = int(elapsed_time_seconds // 3600)
    minutes = int((elapsed_time_seconds % 3600) // 60)
    seconds = int(elapsed_time_seconds % 60)
    if p:
        # Print the elapsed time
        print(f"Elapsed time: {hours} hours, {minutes} minutes, {seconds} seconds")
    elif not p:
        return f"Elapsed time: {hours} hours, {minutes} minutes, {seconds} seconds"

@app.command()
def mc_fractions(mcds: Path, threshold: int):
    open_mcds = MCDS.open(mcds.as_posix(), var_dim="TEs")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cgn = open_mcds["TEs_da"].sel(mc_type="CGN")
        chn = open_mcds["TEs_da"].sel(mc_type="CHN")
        cgnf = cgn.where(cgn.sel(count_type="cov") > threshold)
        chnf = chn.where(chn.sel(count_type="cov") > threshold)


        # here we expected to see a true_divide warning due to cov=0
        raw_frac_g = cgnf.sel(count_type="mc") / cgnf.sel(count_type="cov")
        raw_frac_h = chnf.sel(count_type="mc") / chnf.sel(count_type="cov")
        raw_frac_g = raw_frac_g.fillna(0)
        raw_frac_h = raw_frac_h.fillna(0)
        spCH = raw_frac_h.data.map_blocks(sparse.COO)
        spCG = raw_frac_g.data.map_blocks(sparse.COO)
        mmwrite(mcds.as_posix() + ".ch.mtx", coo_matrix(spCH.compute().to_scipy_sparse()))
        mmwrite(mcds.as_posix() + ".cg.mtx", coo_matrix(spCG.compute().to_scipy_sparse()))


@app.command()
def run_analysis(bed_intersect: Path, sample_name: str, cell_barcodes: Annotated[Path, typer.Argument()]):
    load_bed_intersect = BedTool(bed_intersect)
    if cell_barcodes:
        barcodes_df = pd.read_csv(
            cell_barcodes, sep="\t", names=["barcode"]
        )
        barcodes = barcodes_df["barcode"].tolist()
    else:
        barcodes = None
    (
        unique_TEs_sparse_matrix,
        TE_Fams_sparse_matrix,
        unique_TEs_list,
        TEs_fams,
        barcode_dict,
    ) = make_cell_x_element_matrix(load_bed_intersect, barcodes)
    unique_TEs_sparse_matrix = convert_df_to_sparse(unique_TEs_sparse_matrix)
    TE_Fams_sparse_matrix = convert_df_to_sparse(TE_Fams_sparse_matrix)
    print("Matrix(s) created",
    #    display_elapsed_time(start_time, p=False)
          )
    # make the output directory
    output_dir_fam = "TE_Fam_matrix_" + sample_name + "/"
    os.makedirs(output_dir_fam, exist_ok=True)
    # start saving barcode info
    compress_tsv_file("barcodes.tsv.gz", output_dir_fam, barcode_dict)
    # starting saving the features
    fam_df = make_features_df(TEs_fams)
    fam_path = output_dir_fam + "features.tsv.gz"
    save_df_as_gz(fam_df, fam_path)
    # start saving the matrix
    fam_mtx_path = output_dir_fam + "matrix.mtx"
    compress_sparse_matrix(TE_Fams_sparse_matrix, fam_mtx_path)
    print("Files saved to TE_Fams_matrix directory")
    # start on the unique TE info
    output_dir_unique = "TE_Unique_matrix_" + sample_name + "/"
    os.makedirs(output_dir_unique, exist_ok=True)
    # start saving barcode info
    compress_tsv_file("barcodes.tsv.gz", output_dir_unique, barcode_dict)
    # start saving the features
    unique_df = make_features_df(unique_TEs_list)
    unique_feat_path = output_dir_unique + "features.tsv.gz"
    save_df_as_gz(unique_df, unique_feat_path)
    # start saving the matrix
    unique_mtx_path = output_dir_unique + "matrix.mtx"
    compress_sparse_matrix(unique_TEs_sparse_matrix, unique_mtx_path)
    print("Files saved to TE_Unique_matrix directory")
    # Print the elapsed time
    print(
        "info saved to respective directories",
    #    display_elapsed_time(start_time, p=False),
         )
    print("DONE")
