#!/bin/bash

for file in optimize_aceff.json optimize_aimnet2.json optimize_mace.json optimize_sage.json;do sed -i "s/CONFIG_JSON/${file}/g" slurmscript.sh;sbatch slurmscript.sh;sed -i "s/${file}/CONFIG_JSON/g" slurmscript.sh;done

for file in optimize_mace_omol.json optimize_fennix_bio1_ion.json optimize_mace_off24_medium.json;do sed -i "s/CONFIG_JSON/${file}/g" slurmscript.sh;sbatch slurmscript.sh;sed -i "s/${file}/CONFIG_JSON/g" slurmscript.sh;done

for file in optimize_fennix_bio1_ion.json;do sed -i "s/CONFIG_JSON/${file}/g" slurmscript.sh;sbatch slurmscript.sh;sed -i "s/${file}/CONFIG_JSON/g" slurmscript.sh;done
