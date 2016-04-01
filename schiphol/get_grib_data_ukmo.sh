#!/bin/bash
# TODO Make coordinates argument
# Model grid point definition at Schiphol
gridLat=52.31555938720703
gridLong=4.790283679962158

# TODO Make argument
gribdir="/data/archive/model/ukmo/"

basedir="./schiphol/"
paramdir=${basedir}data/
datadir=${basedir}data/grib/

export GRIB_INVENTORY_MODE=time

readLines(){
    old_IFS=$IFS
    array=()
    while IFS= read -r line; do
        array+=("$line")    # Append to array
    done < $2
    # echo ${array[*]}
    eval $1=\("${array[@]}"\) # Reassign array
    IFS=$old_ifs
}

processFile_ukmo(){
    old_IFS=$IFS
    # Argument 1: input file
    input_file=${1}
    # Argument 2: input variable
    input_variable=${2}
    # Extract grib data
    lines=`grib_ls -p ${col_headers} -w indicatorOfParameter:d=input_variable,level:d=0 -l ${gridLat},${gridLong},4 ${input_file}`
    # Compress spaces, remove trailing whitespace and header line.
    lines=`echo "${lines}" | tr -s ' ' | sed 's/[ \t]*$//' | tail -n +2`

    # Count number of lines with actual data.
    # The first line containing a file name is the file summary and marks the
    # last line of data in the file.
    nr_data_lines=`echo "${lines}" | awk '/.grb/{ print NR; exit }'`

    # Number of data lines with
    data_lines=`echo "${lines}" | head -n $((${nr_data_lines}-1))`
    data_lines=`echo "${data_lines}" | tr ' ' ','` # Covnert data lines into csv

    meta_lines=`echo "${lines}" | tail -n 4`
    unset lines # Free memory
    IFS=$old_IFS
}

writeData(){
    # Argument 1: file suffix
    suffix=${1}

    # Argument 2: data lines to be written to file.
    # Argument 3: meta lines to be written to file.
    echo "Writing `echo ${2} | wc -l` lines to data file."
    echo "${2}" > "${datadir}data_${suffix}.csv" # Write data lines to file.
    echo "${3}" > "${datadir}meta_${suffix}.tmp" # Write data lines to file.
}

if [ ! -d "$datadir" ]; then
    mkdir $datadir
fi

# Read parameters from file.
readLines "headers_from_file" "${paramdir}data_headers_ukmo.txt"
nr_columns=${#headers_from_file[@]}
col_headers=$(IFS=$','; echo "${headers_from_file[*]}")
echo "$nr_columns column headers read for processing."

readLines "dates" "${paramdir}dates.txt"
nr_dates=${#dates[*]}
echo "$nr_dates dates read for processing."

readLines "forecast_hours" "${paramdir}forecast_hours.txt"
nr_forecast_hours=${#forecast_hours[*]}
echo "$nr_forecast_hours forecast hours read for processing."

readLines "variables" "${paramdir}variable_ids_ukmo.txt"
nr_variables=${#variables[*]}
echo "$nr_variables variables read for processing."
model_prefixes[0]="mmm_ukmo"
nr_models=${#model_prefixes[@]}

# Model issues
model_issues[0]=0
model_issues[1]=12
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

            full_data=()
            full_file_suffix=${model_prefix}_${variable}_${issue_time}
            if [ "${model_prefix}" == "mmm_ukmo" ]
            then
                # Loop over dates
                for ((i=0; i<${nr_dates}; i++)); do
                    date=${dates[i]}
                    
                    # Loop over forecast hours
                    for ((h=0; h<${nr_forecast_hours}; h++)); do
                      forecast_hour=${forecast_hours[h]}
                      
                      # Loop core for ukmo
                      file_path=${gribdir}${model_prefix}_${date}${issue_time}_${forecast_hour}.grb
                      echo ${file_path}
                      file_count=$((file_count+1))
                      # Process file: sets ${data_lines}
                      #processFile_ukmo ${file_path} ${variable}
                      echo "Processed ${file_count} files.."
                      # Concatenate data
                      # Check if full_data is empty
                      #if [ ${#full_data[@]} -eq 0 ]; then
                          # Set full_data equal to data_lines
                          #full_data=${data_lines}
                      #else
                          # Concatenate data_lines to full_data
                          #data_lines=`echo "${data_lines}" | tail -n +2` # Strip off header line.
                          #full_data=`echo -e "${full_data}\n${data_lines}"`
                      #fi
                done
            fi

            # Write data to file
            writeData "${full_file_suffix}" "${full_data}" "${meta_lines}"
        done
    done
done
echo "Finished."
