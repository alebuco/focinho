import dlib

opcoes = dlib.shape_predictor_training_options()
dlib.train_shape_predictor("positivas_manual/cara_cachorro.xml", "positivas_manual/detector_cara_cachorro_ccl.dat", opcoes)