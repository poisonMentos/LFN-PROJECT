@echo off
echo Running the Final_script_uncommented.py with figures mode...
py Final_script_uncommented.py network_data network_figures figures

echo Running the Final_script_uncommented.py with stat mode...
py Final_script_uncommented.py network_data network_stat_data stat

echo Running the Final_script_uncommented.py with data mode...
py Final_script_uncommented.py network_data network_dataframe data

echo All commands executed successfully.
pause