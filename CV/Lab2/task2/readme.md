# Task 2 - CT and MRI fusion

Put one correctly matched CT/MRI slice pair in `data/`, then run:

```powershell
python modal_fusion.py --ct data/ct_slice.png --mri data/mri_slice.png


The two images must represent the same aligned heart slice. The script resizes only for a display-compatible matrix; it cannot register a mismatched pair.
