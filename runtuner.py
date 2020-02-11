from genetictuner import GeneticTuner
from genetictuner import ModelTypes

#Create a tuner.
tuner = GeneticTuner(ModelTypes.TYPE_MLP)

#Images must be loaded to tuner before running other methods.
tuner.load_images("/adept_data/new_buoy_imgs") #The directory path starts from home ~
tuner.display_batch()

tuner.initialize_population(population_size=5, epochs=2)
tuner.run_tuner()