@echo off
echo Running the Final_script_uncommented.py with figures mode...
py Final_script_uncommented.py test_data test_figures figures

echo Running the Final_script_uncommented.py with stat mode...
py Final_script_uncommented.py test_data test_stat_data stat

echo Running the FFinal_script_uncommented.py with data mode...
py Final_script_uncommented.py test_data test_dataframe data

echo All commands executed successfully.
pause