CPSC 8430 HW4
Implementation of a DCGAN, WGAN, and ACGAN

To run, place the desired main script "HW4_XXGAN_v2" 
in the same directory as the file with its supplemental functions "HW4_XXGAN_Supp_v2"

Running the script will train the model on the Cifar-10 dataset, downloading it if necessary

Once training is completed, the performance will be evaluated by creating two subfolders
"real_images" - Directly sampled from Cifar-10
"generated_images_fid" - Images sampled from trained generator
and calculate the FID score
