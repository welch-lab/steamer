version 1.0
workflow run_full {
  input {
    File fullin_TEs
    Array[String] file_id
    Array[File] allcs
    Array[Int]  size_col
    File chrom_size
    Array[String] fullin_sample_name_column
    Int memory_GB
    Int nCPUs
    Int threshold_QC
  }
  String fullin_sample_name = fullin_sample_name_column[0]
    parameter_meta {
      fullin_TEs: "Path to bed file containing TEs"
      file_id: "terra table column containing file IDs"
      allcs: "terra table column containing location of 'allc_*.tsv.gz'"
      size_col: "terra column containing sizes of allc files"
      chrom_size: "Path to chrom.sizes obtained by UCSC fetchChromSizes.sh"
      fullin_sample_name_column: "name of sample"
      memory_GB: "memory, in gigabytes"
      nCPUs: "CPUs for parallel execution"
      threshold_QC: "Threshold for discarding methylation value"
  }
  call sum {
    input:
      sizes = size_col
  }
  call mangle_bed {
    input:
      bed = fullin_TEs,
      mem = memory_GB
  }
  call generate_dataset {
    input:
        fileIDs = file_id,
        allc_list = allcs,
        SampleName = fullin_sample_name,
        nCPU = nCPUs,
        DiskSize = sum.size,
        mangledTEs = mangle_bed.bed_mangled,
        chromSize = chrom_size,
        mem = memory_GB
  }
  call calculate_fractions {
    input:
        tempzarr = generate_dataset.zarrTar,
        SampleName = fullin_sample_name,
        mem = memory_GB,
        threshold = threshold_QC
  }
  output {
    File mtx_ch = calculate_fractions.count_mat_ch
    File mtx_cg = calculate_fractions.count_mat_cg
  }
}

task mangle_bed {
    input {
        File bed
        Int mem
    }
    command <<<
      run_steamer mangle-bed-file-ids ~{bed} TEs_mangled.bed
    >>>
    output {
        File bed_mangled = "TEs_mangled.bed"
    }
    runtime {
    docker: "quay.io/welch-lab/steamer:latest"
    memory: mem + "GB"
  }
}

task generate_dataset {
    input {
        Array[String] fileIDs
        Array[File]   allc_list
        String SampleName
        Int nCPU
        Int DiskSize
        File mangledTEs
        File chromSize
        Int mem
    }
    parameter_meta {
        allc_list: {
            description: "Terra table column containing location of 'allc_*.tsv.gz",
            #localization_optional: true
        }
    }
    Int nCPUscale = ceil(nCPU*0.75)
    Int disk = DiskSize + 375
    String disk_string = "local-disk " + disk + " LOCAL"
    Array[Array[String]] initial_paired = [fileIDs, allc_list]
    Array[Array[String]] tsvPaired = transpose(initial_paired)
    File allc_table = write_tsv(tsvPaired)
    command <<<
    #CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    #GCS_REQUESTER_PAYS_PROJECT=$(curl -s "http://metadata.google.internal/computeMetadata/v1/project/project-id" -H "Metadata-Flavor: Google") \
    #GCS_OAUTH_TOKEN=$(gcloud auth application-default print-access-token) \
    sed -i 's;gs://;/cromwell_root/;g' ~{allc_table}; \
    allcools generate-dataset --allc_table ~{allc_table} --output_path=~{SampleName}.mcds --obs_dim cell \
    --cpu ~{nCPUscale} --chunk 50 --regions TEs ~{mangledTEs} --chrom_size_path ~{chromSize} \
    --quantifiers TEs count CGN,CHN; tar -cf tempzarr.tar ~{SampleName}.mcds
    >>>
    output {
#this should be a Directory but cromwell doesn't support WDL 1.2
        File zarrTar = "tempzarr.tar"
    }
    runtime {
      docker: "quay.io/welch-lab/steamer:latest"
      memory: mem + "GB"
      cpu: nCPU
      disks: disk_string
    }
}

task calculate_fractions {
    input {
        File tempzarr
        String SampleName
        Int mem
        Int threshold
    }
    command <<<
    tar -xf ~{tempzarr}; \
    run_steamer mc-fractions ~{SampleName}.mcds ~{threshold}
    >>>
    output {
        File count_mat_ch = SampleName + ".mcds.ch.mtx"
        File count_mat_cg = SampleName + ".mcds.cg.mtx"
    }
    runtime {
        docker: "quay.io/welch-lab/steamer:latest"
        memory: mem + "GB"
  }
}


task sum {
  input {
    Array[Int] sizes
  }
  command <<<
  python3 <<CODE
  print(sum([~{sep=',' sizes}])>>30)
  CODE
  >>>
  output {
    Int size = ceil(read_float(stdout()))
  }
  runtime {
    docker: "python:3.12"
  }
}
