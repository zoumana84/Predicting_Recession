#define a flag for each hyper parameter you want to fine tune and a default value for that parameter
FLAGS  = flags(flag_numeric("nodes1", 4),
               flag_numeric("drop_out1", 0.3), #this is a float between 0 and 1
               flag_numeric("nodes2", 4),
               flag_numeric("drop_out2", 0.3),
               flag_numeric("batch_size", 100),
               flag_string("activation1", "relu"),
               flag_string("activation2", "sigmoid"),
               flag_numeric("learning_rate", 0.01),
               flag_numeric("epochs", 30)
) 

model = keras_model_sequential() %>%
  layer_dense(units = FLAGS$nodes1, activation = FLAGS$activation1, 
              input_shape = dim(recession_trainScaled)[2]) %>%
  layer_dropout(rate = FLAGS$drop_out1)%>%
  layer_dense(units = FLAGS$nodes2, activation = FLAGS$activation1) %>%
  layer_dropout(rate = FLAGS$drop_out2)%>%
  layer_dense(units = 1, activation = FLAGS$activation2)

model %>% compile(
  optimizer = optimizer_adam(lr=FLAGS$learning_rate), 
  loss = 'binary_crossentropy',
  metrics = c('AUC'))

model %>% fit(recession_trainScaled, recession_trainScaledlabel, 
              epochs = FLAGS$epochs, 
              batch_size= FLAGS$batch_size, 
              validation_data = list(recession_valScaled, recession_valScaledlabel))
