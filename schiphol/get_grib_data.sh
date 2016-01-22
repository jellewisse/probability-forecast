#!/bin/bash

# TODO Make coordinates argument
# Model grid point definition at Schiphol
gridLat=52.31555938720703
gridLong=4.790283679962158

gribdir="/data/research/verification/data_requests/requests_2015/bcs30429/data/" # TODO Make argument

basedir="./schiphol/"
paramdir=${basedir}data/
tmpdir=${basedir}data/tmp/

export GRIB_INVENTORY_MODE=time

readLines(){
    array=()
    while IFS= read -r line; do
        array+=("$line")    # Append to array
    done < $2
    # echo ${array[*]}
    eval $1=\("${array[@]}"\) # Reassign array
}

processFile(){
    # Argument 1: input file
    input_file=${1}

    # Argument 2: file suffix
    suffix=${2}

    lines=`grib_ls -p ${col_headers} -w level:d=0 -l ${gridLat},${gridLong},4 ${input_file}` # Extract grib data
    lines=`echo "${lines}" | tr -s ' ' | sed 's/[ \t]*$//' | tail -n +2`  # Compress spaces, remove trailing whitespace and header line.

    nr_data_lines=`echo "${lines}" | awk '/.grb/{ print NR; exit }'`
    data_lines=`echo "${lines}" | head -n $((${nr_data_lines}-1))`
    meta_lines=`echo "${lines}" | tail -n 4`
    unset lines # Free memory

    data_lines=`echo "${data_lines}" | tr ' ' ','` # Covnert data lines into csv
    echo "${data_lines}" > "${tmpdir}data_${suffix}.csv" # Write data lines to file.
    echo "${meta_lines}" > "${tmpdir}meta_${suffix}.tmp" # Write data lines to file.
}

# Read parameters from file.
readLines "headers_from_file" "${paramdir}data_headers.txt"
nr_columns=${#headers_from_file[@]}
col_headers=$(IFS=$','; echo "${headers_from_file[*]}")
echo "$nr_columns column headers read for processing."

readLines "dates" "${paramdir}dates.txt"
nr_dates=${#dates[*]}
echo "$nr_dates dates read for processing."

readLines "variables" "${paramdir}variable_ids.txt"
nr_variables=${#variables[*]}
echo "$nr_variables variables read for processing."

model_prefixes[0]="eps"
model_prefixes[1]="control"
model_prefixes[2]="fc"
nr_models=${#model_prefixes[@]}

# Model issues
model_issues[0]=0
model_issues[1]=1200
nr_issues=${#model_issues[@]}

file_count=0
echo "Starting main loop.."
# Loop over different models
for((m=0; m<${nr_models}; m++)); do
    model_prefix=${model_prefixes[m]}

    # Loop over model issues
    for((k=0; k<${nr_issues}; k++)); do
        issue_time=${model_issues[k]}

        # Loop over elements
        for((j=0; j<${nr_variables}; j++)); do
            variable=${variables[j]}

            if [ "${model_prefix}" != "eps" ]
            then

                file_suffix=${model_prefix}_${variable}_${issue_time}
                processFile "${gribdir}bcs30429_${file_suffix}.grb" "${file_suffix}"
                file_count=$((file_count+1))
                echo "Processed ${file_count} files.."
            else
                # Loop over dates
                for ((i=0; i<${$}; i++)); do
                    date=${dates[i]}
                    file_suffix=${model_prefix}_${date}_${variable}_${issue_time}
                    processFile "${gribdir}bcs30429_${file_suffix}.grb" "${file_suffix}"
                    file_count=$((file_count+1))
                    echo "Processed ${file_count} files.."
                done
            fi
        done
    done
done
echo "Finished."
