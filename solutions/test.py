split_data_into_train_test_val()
model.make_graph()
hyperparameters = random_search?()

if restoring_training:
    load_weights()
else:
    initialize_weights()

for hyperparameter_set in hyperparameters:
    #train
    while val_loss is decreasing and epoch < max_epoch:
        #train on an epoch
        for b in num_batches:
            batch = dataset.get_next_batch()
            batch = augment(batch)
            loss, accuracy = run(train_step, loss, accuracy)
            log(loss, accuracy, etc.)
        epoch+=1
        val_batch = dataset.get_next_val_batch()
        loss, accuracy = run(loss, accuracy)
        log(loss, accuracy, etc.)



