version 1.0
workflow runsteamer {
  input {
    File? in_bedfile
    File? in_barcode_list
    String in_sample_name
    Int in_mem
  }
  parameter_meta {
    in_bedfile: "Path to intersected bedfile"
    in_barcode_list: "Path to barcode list"
    in_sample_name: "name of sample"
    in_mem: "Memory in GB"
  }
  call run_analysis {
    input:
      bedfile = in_bedfile,
      barcode_list = in_barcode_list,
      sample_name = in_sample_name,
      mem = in_mem
  }
  output {
    File FamMat = run_analysis.FamMat
    File UniqueMat = run_analysis.UniqueMat
    File UniqueDF = run_analysis.UniqueDF
    File UniqueBar = run_analysis.UniqueBar
    File FamDF = run_analysis.FamDF
    File FamBar = run_analysis.FamBar
  }
}

task run_analysis {
  input {
    File? bedfile
    File? barcode_list
    String sample_name
    Int mem
  }
  command <<<
    run_steamer run-analysis ~{bedfile} ~{sample_name} ~{barcode_list}
  >>>
    output {
    File FamMat = "TE_Fam_matrix_" + sample_name + "/matrix.mtx.gz"
    File UniqueMat = "TE_Unique_matrix_" + sample_name + "/matrix.mtx.gz"
    File UniqueDF = "TE_Unique_matrix_" + sample_name + "/features.tsv.gz"
    File UniqueBar = "TE_Unique_matrix_" + sample_name + "/barcodes.tsv.gz"
    File FamDF = "TE_Fam_matrix_" + sample_name + "/features.tsv.gz"
    File FamBar = "TE_Fam_matrix_" + sample_name + "/barcodes.tsv.gz"
  }
  runtime {
    docker: "quay.io/welch-lab/steamer:latest"
    memory: mem + "GB"
  }
}
