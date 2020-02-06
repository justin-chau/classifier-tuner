from paramtuner import ParamTuner
from paramtuner import ModelTypes
from paramtuner import TunerTypes

#Create a tuner.
tuner = ParamTuner(ModelTypes.TYPE_MLP, TunerTypes.TYPE_GENETIC)

#Images must be loaded to tuner before running other methods.
tuner.load_images("/adept_data/new_buoy_imgs") #The directory path starts from home ~


print("------------------IMAGE COUNT------------------", end="\n\n")
print(tuner.get_image_count(), end="\n\n")
print("-----------------------------------------------", end="\n\n")
print("------------------CLASS NAMES------------------", end="\n\n")
print(tuner.get_class_names(), end="\n\n")
print("-----------------------------------------------", end="\n\n")

tuner.display_batch()

tuner.initialize_population(population_size=3, n_epochs=1)
tuner.run_tuner()