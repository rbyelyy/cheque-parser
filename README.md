# README #

What this project is doing?

1. Get photo of any kind of bills and parse text on it 
(opencv + tesseract)

2. Automatically submit successfully parsed bill into 
accounting government system
(selenium)

3. Cracking 'CAPTCHA' for applying submitting process on continues manner

4. Process is self recovered and have very high survival level


### Preconditions ###

* Python > 2.7 
* OpenCV 3.X python package
* Firefox > 45

### How to run? ###
POST BILLS:
python post_bills.py

IMAGE PROCESSING:
With  'Adaptive Threshold'

python image_processing.py -src source_folder/1.jpg -vc 4 -hc 5 -bt 110 -el 14 -tt harrison -hl 270 --adaptive_threshold

Without 'Adaptive Threshold'

python image_processing.py -src source_folder/1.jpg -vc 4 -hc 5 -bt 110 -el 14 -tt harrison -hl 270 --no-adaptive_threshold

Without clean (by default 'clean')

python image_processing.py -src source_folder/1.jpg -ci not-clean -vc 4 -hc 5 -bt 110 -el 14 -tt harrison -hl 270 --adaptive_threshold

