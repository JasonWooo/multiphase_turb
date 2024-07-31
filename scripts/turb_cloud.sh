# go to working directory
source ~/.zshrc
cd $DP/MP_multiphase_gas/turb_240531_2/cloud
pwd

# introduce and evolve the cold core
$ATHENA_DIR/bin/athena -i athinput_cloud.turb -r ../turb/Turb.final.rst
echo "\n\n\n\n"
echo "Done!"
cd $ATHENA_DIR