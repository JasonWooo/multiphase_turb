# go to working directory
source ~/.zshrc
cd $DP/MP_multiphase_gas/turb_240531_2/turb
pwd

# generate the turbulence
$ATHENA_DIR/bin/athena -i athinput_init.turb
echo "\n\n\n\n"
echo "Turbulence introduced"
cd $ATHENA_DIR