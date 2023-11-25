import csv
import numpy as np
import sdss
from sdss.photometry import frame_filename, obj_frame_url, \
     download_file, unzip, get_df, df_radec2pixel

# Open the CSV file
with open('ObjectList.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip the header row

# Open the text file for writing
    with open('galaxyDataset.txt', 'w') as out_file:
        # Loop through each row in the CSV file
        for row in reader:
            # Get the object ID from the first column
            object_id = int(row[0])
            ph = sdss.PhotoObj(object_id)
            data = ph.cutout_image()
            classification = ph.type
            # Flatten the image data and save it to the text file
            np.savetxt(out_file, data.flatten(), newline=' ', fmt='%s')
            out_file.write(f' {classification}\n')


#Testing writing with just a single object
object_id = 1237662662683918640
ph = sdss.PhotoObj(object_id)
data = ph.cutout_image()
classification = ph.type
ph.show()
with open('galaxyDataset.txt', 'w') as out_file:

    np.savetxt(out_file, data.flatten(), newline=' ', fmt='%s')
    out_file.write(f' {classification}\n')