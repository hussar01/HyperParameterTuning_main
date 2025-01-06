
#---------------------------------------------------------------------------------------
script inference_training.py
#-----------------------------------------------------------------------------------------
def do_train(trial, config, model: t.nn.Module, loader_train: DataLoader, loader_valid: DataLoader, epochs: int = 1,
             device: t.types.Device = CONFIG.DEVICE, early_stop=False, fold_idx=-1, tensorboard=False, tensor_folder=None,
             multi_feature=False):
   
    model.train()
    # Init Loss Function + Optimizer with Learning Rate Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=CONFIG.MI.LR.milestones,
                                                  gamma=CONFIG.MI.LR.gamma)

    # Tensorbaord initialization + Create folder to store tensorboard related data
    if tensorboard:
        fold_folder = tensor_folder + "/fold" + fold_idx.__str__()
        os.mkdir(fold_folder)
        tb = SummaryWriter(fold_folder)

    print("###### Training started")
    loss_values_train, loss_values_valid = np.full((epochs), fill_value=np.inf), np.full((epochs), fill_value=np.inf)
    best_epoch = 0
    best_model = model.state_dict().copy()

    for epoch in range(epochs):
        print(f"## Epoch {epoch} ")
        model.train()
        running_loss_train, running_loss_valid = 0.0, 0.0
        # Wrap in tqdm for Progressbar in Console
        pbar = tqdm(loader_train, file=sys.stdout)
        # Training in batches from the DataLoader
        for idx_batch, (*inputs, labels) in enumerate(pbar):
            inputs, labels = inputs, labels.long().to(device)   # Convert to correct types + put on GPU
            optimizer.zero_grad()
            # zero the parameter gradients
            if multi_feature:
                outputs = model(inputs)  # forward + backward + optimize
            else:
                outputs = model(inputs[0])
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss_train += loss.item()
        pbar.close()

        # Loss of entire epoch / amount of batches
        epoch_loss_train = (running_loss_train / len(loader_train))
        if tensorboard:
            tb.add_scalar("epoch_loss_train", epoch_loss_train, epoch)

        if loader_valid is not None:
            # Validation loss on Test Dataset
            # if early_stop=True: Used to determine best model state
            with torch.no_grad():
                model.eval()
                for idx_batch, (*inputs, labels) in enumerate(loader_valid):
                    # Convert to correct types + put on GPU
                    inputs, labels = inputs, labels.long().to(device)

                    # forward
                    if multi_feature:
                        outputs = model(inputs)
                    else:
                        outputs = model(inputs[0])
                    # print("out",outputs.shape,"labels",labels.shape)
                    loss = criterion(outputs, labels)

                    running_loss_valid += loss.item()
            epoch_loss_valid = (running_loss_valid / len(loader_valid))
            if tensorboard:
                tb.add_scalar("epoch_loss_valid", epoch_loss_valid, epoch)
            # Determine if epoch (validation loss) is lower than all epochs before -> work best model

            # Report metrics to Optuna at each epoch for intermediate evaluation
            trial.report(epoch_loss_valid, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            
            if epoch_loss_valid < loss_values_valid.min():
                best_model = model.state_dict().copy()
                best_epoch = epoch
            print('[%3d] Training loss/batch: %f\tTesting loss/batch: %f' %
                (epoch, epoch_loss_train, epoch_loss_valid))
            loss_values_valid[epoch] = epoch_loss_valid
        else:
            print('[%3d] Training loss/batch: %f' % (epoch, epoch_loss_train))
        loss_values_train[epoch] = epoch_loss_train

        lr_scheduler.step()

    if tensorboard:
        tb.close()
    print("Training finished ######")

    return loss_values_train, loss_values_valid, best_model, best_epoch




#---------------------------------------------------------------------------------------
script cross_fold_vaalidation01.py
#-----------------------------------------------------------------------------------------
def cross_fold_validation_2(subjects, save_results=True, save_all=True, tensorboard=True,trial=trial):
    # Load and preprocess data (same as before)
    device = CONFIG.DEVICE
    ds_name = str(CONFIG.EEG.DS_NAME)
    dataset = DATASETS[ds_name]
    now = datetime.now()
    now_string = datetime_to_folder_str(now)
    
    # Create results folder if needed
    folder = "none"
    if save_results:
        folder = f"{results_cross_fold_validation}/{CONFIG.EEG.MODE}_{CONFIG.NET.NET_TYPE}_{CONFIG.EEG.DS_NAME}_{now_string}"
        print("- Results will be stored in folder:", folder)
        makedir(folder)
    if tensorboard:
        tensor_folder = f"{results_cross_fold_validation}/tensorboard/{CONFIG.EEG.MODE}/{CONFIG.NET.NET_TYPE}_{CONFIG.EEG.DS_NAME}_{now_string}"
        makedir(tensor_folder)

    # Load all complete trials from the dataset
    preloaded_data, preloaded_labels = load_trials( 
        CONFIG.EEG.DS_NAME, subjects,
        CONFIG.MI.N_ACTIVE_CLASSES,
        CONFIG.EEG.TRIAL_START, CONFIG.EEG.TRIAL_END,
        verbose=VERBOSE
    )

    # Combine axis 0 and 1
    preloaded_data = preloaded_data.reshape(
        (preloaded_data.shape[0] * preloaded_data.shape[1], preloaded_data.shape[2], preloaded_data.shape[3])
    )
    preloaded_labels = preloaded_labels.reshape(
        (preloaded_labels.shape[0] * preloaded_labels.shape[1])
    )

    # Define hyperparameters using trial from optuna

    if trial is not None:
        config = {
            "batch_size": trial.suggest_categorical("batch_size", [32, 64]),
            "lr": trial.suggest_categorical("lr", [1e-3, 2e-4, 2e-3]),
            "lrm": trial.suggest_categorical("lrm", [(20, 50), (10, 30, 60)]),
            "lrg": trial.suggest_categorical("lrg", [0.1]),
            # Add other hyperparameter suggestions here
        }

    # Assign the suggested hyperparameters to CONFIG
    CONFIG.MI.BATCH_SIZE = config["batch_size"]
    CONFIG.MI.LR.start = config["lr"]
    CONFIG.MI.LR.milestones = config["lrm"]
    CONFIG.MI.LR.gamma = config["lrg"]
    # Assign other hyperparameters similarly


    # Split into outer 1 training and test sets
    outer_skf = StratifiedKFold(n_splits=1, shuffle=True, random_state=0)
    trainval_indices, test_indices = next(outer_skf.split(preloaded_data, preloaded_labels))

    trainval_data = preloaded_data[trainval_indices]
    trainval_labels = preloaded_labels[trainval_indices]
    test_data = preloaded_data[test_indices]      #outer fold test set not used
    test_labels = preloaded_labels[test_indices]  #outer fold labels not used

    # Inner cross-validation setup
    INNER_FOLDS = 5  # Define the number of inner folds
    inner_skf = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=0)
    inner_fold_val_losses = []

    # Initialize run_data for tracking
    folds = INNER_FOLDS
    run_data = MLRunData(folds, CONFIG.MI.N_CLASS, CONFIG.MI.EPOCHS, folds)
    run_data.start_run()
    
    # Inner cross-validation loop
    for inner_fold_idx, (inner_train_indices, val_indices) in enumerate(inner_skf.split(trainval_data, trainval_labels)):
        print(f"Inner Fold {inner_fold_idx + 1}/{INNER_FOLDS}")

        # Split into inner training and validation sets
        inner_train_data = trainval_data[inner_train_indices]
        inner_train_labels = trainval_labels[inner_train_indices]
        val_data = trainval_data[val_indices]
        val_labels = trainval_labels[val_indices]

        # Preprocess data
        train_feature, train_label = pre_process_data(
            data=inner_train_data, labels=inner_train_labels,
            n_active_classes=CONFIG.MI.N_ACTIVE_CLASSES,
            rest_class=CONFIG.MI.REST_CLASS,
            trial_start=CONFIG.EEG.TRIAL_START, trial_end=CONFIG.EEG.TRIAL_END,
            cue_on=CONFIG.EEG.CUE_ON, cue_off=CONFIG.EEG.CUE_OFF,
            ts_size=CONFIG.EEG.TS_SIZE, ts_offset=CONFIG.EEG.TS_OFFSET,
            scaling=CONFIG.FILTER.FEATURE_SCALING, equalize=CONFIG.EEG.EQUALIZE_TSLICES_PER_CLASS,
            verbose=VERBOSE, ignore_overlap_slices=True
        )
        test_feature, test_label = pre_process_data(
            data=val_data, labels=val_labels,
            n_active_classes=CONFIG.MI.N_ACTIVE_CLASSES,
            rest_class=CONFIG.MI.REST_CLASS,
            trial_start=CONFIG.EEG.TRIAL_START, trial_end=CONFIG.EEG.TRIAL_END,
            cue_on=CONFIG.EEG.CUE_ON, cue_off=CONFIG.EEG.CUE_OFF,
            ts_size=CONFIG.EEG.TS_SIZE, ts_offset=CONFIG.EEG.TS_OFFSET,
            scaling=CONFIG.FILTER.FEATURE_SCALING, equalize=CONFIG.EEG.EQUALIZE_TSLICES_PER_CLASS,
            verbose=VERBOSE, ignore_overlap_slices=True
        )

        # Rearrangement
        train_feature, train_label = re_arrange_data1(
            feature=train_feature, label=train_label,
            net_type=CONFIG.NET.NET_TYPE, n_class=CONFIG.MI.N_CLASS,
            verbose=VERBOSE,
            transform_to_two_D=dataset.transform_to_two_D
        )
        test_feature, test_label = re_arrange_data1(
            feature=test_feature, label=test_label,
            net_type=CONFIG.NET.NET_TYPE, n_class=CONFIG.MI.N_CLASS,
            verbose=VERBOSE,
            transform_to_two_D=dataset.transform_to_two_D
        )

        n_channels = train_feature.shape[2]  # Number of EEG channels
        n_maps = train_feature.shape[1]

        # Create DataLoaders
        inner_train_loader = create_data_loader(
            data=train_feature, labels=train_label,
            device=device, batch_size=config["batch_size"], verbose=VERBOSE
        )
        val_loader = create_data_loader(
            data=test_feature, labels=test_label,
            device=device, batch_size=config["batch_size"], verbose=VERBOSE
        )


        # Initialize model
        model = EEGNet(N=CONFIG.MI.N_CLASS, T=CONFIG.EEG.TS_SIZE * CONFIG.EEG.SAMPLERATE, C=inner_train_feature.shape[2])
        model.to(device)

        # Train the model on inner training set and validate
        train_results = do_train(
            trial=trial,
            config=config,
            model=model,
            loader_train=inner_train_loader,
            loader_valid=val_loader,
            epochs=CONFIG.MI.EPOCHS,
            device=device,
            early_stop=False,
            fold_idx=inner_fold_idx + 1,
            tensorboard=tensorboard,
            tensor_folder=tensor_folder
        )
        run_data.set_train_results(inner_fold_idx, train_results)
        
        # Extract validation loss directly from train_results
        _, loss_values_valid, _, _ = train_results
        # Collect final validation loss
        final_val_loss = loss_values_valid[-1]  # Last epoch'S validation loss
        inner_fold_val_losses.append(final_val_loss)
        print(f"Validation Loss for Inner Fold {inner_fold_idx + 1}: {final_val_loss:.4f}")
        
        # Evaluate on validation set and collect results
        #acc, act_labels, pred_labels = do_test(model=model, data_loader=val_loader)
        #run_data.set_test_results(inner_fold_idx, acc, act_labels, pred_labels)
        #run_data.end_run()

        #inner_fold_idx = inner_fold_idx + 1 //no need to manually increase inner fold because of ennumerate

    # Compute mean validation loss across inner folds
    mean_val_loss = np.mean(inner_fold_val_losses)
    print(f"Mean Validation Loss: {mean_val_loss:.4f}")

    # Return folder, run_data, and mean_val_loss
    return folder, run_data, mean_val_loss



def objective(trial):
    # Call cross_fold_validation_2 and extract mean_val_loss
    _, _, mean_val_loss = cross_fold_validation_2(
        subjects=subjects,
        save_results=False,
        save_all=False,
        tensorboard=False,
        trial=trial,
    )
    # Return mean validation loss for optimization
    return mean_val_loss

def run_optuna():
    start_time = time.time()
    # Define constants

    MIN_RESOURCE = 1
    REDUCTION_FACTOR = 3
    N_STARTUP_TRIALS = 5
    TOTAL_TRIALS = 20  # Set the total number of trials

    # Create Optuna study
    study = optuna.create_study(
        storage="sqlite:///grid_search_3.db",
        study_name="GridSearch_withASHA",
        load_if_exists=False,
        direction="minimize",
        pruner=optuna.pruners.SuccessiveHalvingPruner(
            min_resource=MIN_RESOURCE,
            reduction_factor=REDUCTION_FACTOR
        ),
        sampler=optuna.samplers.TPESampler(
            seed=42,
            n_startup_trials=N_STARTUP_TRIALS,
            multivariate=True,
            group=True
        ),
    )

    # Optimize
    study.optimize(objective, n_trials=TOTAL_TRIALS)

    # Get the best trial
    best_trial = study.best_trial
    print(f"Best hyperparameters: {best_trial.params}")

    # Now retrain on the entire outer training set with the best hyperparameters
    # and evaluate on the outer test set
    # Update CONFIG with best hyperparameters
    CONFIG.MI.BATCH_SIZE = best_trial.params['batch_size']
    # Update other hyperparameters similarly

    # Load and preprocess the entire outer training set and outer test set
    # (Repeat the data loading and preprocessing steps)
     # Load and reshape data
    preloaded_data, preloaded_labels = load_trials(
        CONFIG.EEG.DS_NAME, subjects,
        CONFIG.MI.N_ACTIVE_CLASSES,
        CONFIG.EEG.TRIAL_START, CONFIG.EEG.TRIAL_END,
        verbose=VERBOSE
    )
    preloaded_data = preloaded_data.reshape(
        (preloaded_data.shape[0] * preloaded_data.shape[1],
         preloaded_data.shape[2], preloaded_data.shape[3])
    )
    preloaded_labels = preloaded_labels.reshape(
        (preloaded_labels.shape[0] * preloaded_labels.shape[1])
    )
    folds = 1
    run_data = MLRunData(folds, CONFIG.MI.N_CLASS, CONFIG.MI.EPOCHS, folds)
    run_data.start_run()
    fold_idx = 0
     # Split into outer training and test sets (ensure the same split if needed)
    outer_skf = StratifiedKFold(n_splits=1, shuffle=True, random_state=1)
    #trainval_indices, test_indices = next(outer_skf.split(preloaded_data, preloaded_labels))
     # preprocss data: time slicing and optional normalization and equalization
    for train_index, test_index in outer_skf:
        print("Fold %d:" % (fold_idx + 1))

        # Extract fold specified train and test data
        train_data = preloaded_data[train_index]
        train_labels = preloaded_labels[train_index]
        test_data_ct = preloaded_data[test_index]
        test_labels_ct = preloaded_labels[test_index]

        # preprocss data: time slicing and optional normalization and equalization
        train_feature, train_label = pre_process_data(data=train_data, labels=train_labels,
                                                      n_active_classes=CONFIG.MI.N_ACTIVE_CLASSES,
                                                      rest_class=CONFIG.MI.REST_CLASS,
                                                      trial_start=CONFIG.EEG.TRIAL_START, trial_end=CONFIG.EEG.TRIAL_END,
                                                      cue_on=CONFIG.EEG.CUE_ON, cue_off=CONFIG.EEG.CUE_OFF,
                                                      ts_size=CONFIG.EEG.TS_SIZE, ts_offset=CONFIG.EEG.TS_OFFSET,
                                                      scaling=CONFIG.FILTER.FEATURE_SCALING, equalize=CONFIG.EEG.EQUALIZE_TSLICES_PER_CLASS,
                                                      verbose=VERBOSE, ignore_overlap_slices=True)
        test_feature, test_label = pre_process_data(data=test_data_ct, labels=test_labels_ct,
                                                    n_active_classes=CONFIG.MI.N_ACTIVE_CLASSES,
                                                    rest_class=CONFIG.MI.REST_CLASS,
                                                    trial_start=CONFIG.EEG.TRIAL_START, trial_end=CONFIG.EEG.TRIAL_END,
                                                    cue_on=CONFIG.EEG.CUE_ON, cue_off=CONFIG.EEG.CUE_OFF,
                                                    ts_size=CONFIG.EEG.TS_SIZE, ts_offset=CONFIG.EEG.TS_OFFSET,
                                                    scaling=CONFIG.FILTER.FEATURE_SCALING, equalize=CONFIG.EEG.EQUALIZE_TSLICES_PER_CLASS,
                                                    verbose=VERBOSE, ignore_overlap_slices=True)

        # Net specific data/labels re-arrangement
        train_feature, train_label = re_arrange_data1(feature=train_feature, label=train_label,
                                                      net_type=CONFIG.NET.NET_TYPE, n_class=CONFIG.MI.N_CLASS,
                                                      verbose=VERBOSE,
                                                      transform_to_two_D=dataset.transform_to_two_D)
        test_feature, test_label = re_arrange_data1(feature=test_feature, label=test_label,
                                                    net_type=CONFIG.NET.NET_TYPE, n_class=CONFIG.MI.N_CLASS,
                                                    verbose=VERBOSE,
                                                    transform_to_two_D=dataset.transform_to_two_D)

        # BATCH_size mostly has a strong impact on training speed
        print("- BATCH_size: ", best_trial.param["batch_size"])


        n_channels = train_feature.shape[2]  # Number of EEG channels
        n_maps = train_feature.shape[1]
        # feed training and test data into dataloaders
        train_loader = create_data_loader(data=train_feature, labels=train_label,
                                          device=device, batch_size=best_trial.param["batch_size"], verbose=VERBOSE)
        test_loader = create_data_loader(data=test_feature, labels=test_label,
                                         device=device, batch_size=best_trial.param["batch_size"], verbose=VERBOSE)

        # Net architecture and training specific parameters
        #EPOCHS = CONFIG.MI.EPOCHS
        # Initialize model with best hyperparameters
        model = EEGNet(N=CONFIG.MI.N_CLASS, T=CONFIG.EEG.TS_SIZE * CONFIG.EEG.SAMPLERATE, C=train_feature.shape[2])
        model.to(CONFIG.DEVICE)

        # Train the model on the entire outer training set
        # No validation loader since we are training on the full training set
        loss_values_train, loss_values_valid, _, _ = do_train(
            trial=best_trial, 
            config=best_trial.params,
            model=model,
            loader_train=train_loader,
            loader_valid=test_loader,
            epochs=CONFIG.MI.EPOCHS,
            device=CONFIG.DEVICE,
            early_stop=False,
            fold_idx= fold_idx + 1,
            tensorboard=False,
            tensor_folder=None)
        run_data.set_train_results(fold_idx, train_results)

        # Test the model on the outer test set
        test_accuracy, _, _ = do_test(model=model, data_loader=test_loader)
        
    
        # Test overfitting by testing on Training Dataset
        run_data.accuracies_overfitting[fold_idx], _, __ = do_test(model, train_loader)
        print(f"Training Loss {loss_values_train}")
        print(f"Training Loss {loss_values_valid}")
        print(f"Test Accuracy with Best Hyperparameters: {test_accuracy:.4f}")

        end_time = time.time()
        elapsed_time = end_time - start_time
        # Calculate hours, minutes, and seconds
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        print(f"Total time taken for hyperparameter tuning and evaluation: {hours}:{minutes}:{seconds}")

        

#---------------------------------------------------------------------------------------
script ms_main_v01.py
#-----------------------------------------------------------------------------------------

if __name__ == '__main__':
        """
    main implements a jump table which, depending on the command specified by parameter
    CONFIG.OTHER.COMMAND, executes (jumps to the) corresponding script.
    """
    start_time = time.time()
    # Load configuration parameters
    #config_path = '/home/mstrahn/ms_BCI_work01/ML-BCI-main_v210/results/Cross_fold_val/INTRA_SUBJECT_CONV_PHYS_2024-06-10_08_20_18/CONFIG'
    #config_path = '/home/mstrahn/ms_BCI_work01/ML-BCI-main_v210/results/Opt_trial_time_slice_size/CONV_PHYS_ts-sizes0.60-1.00_2024-06-13_14_00_27/CONFIG'

    #config_path = config_defaults  # Load default configuration parameters
    config_path = '/home/ahussain/PycharmProjects/ML-BCI-main_v210/config/ah_work'
    #config_path = '/home/mstrahn/ms_BCI_work01/ML-BCI-main_v210/config/ms_best_CONV'
    #config_path = '/home/mstrahn/ms_BCI_work01/ML-BCI-main_v210/config/ms_best_LSTM_PARALLEL'

    CONFIG.load_params(config_path)
    print(repr(CONFIG))

    # Execute the command specified by the user
    command = CONFIG.OTHER.COMMAND
    if command == 'CROSS_FOLD_VALIDATION':
        if CONFIG.EEG.MODE == 'INTRA_SUBJECT':
            """ Do intra-subject cross-fold validation for the subject specified in CONF.subject """
            subjects = [CONFIG.EEG.ISV_SUBJECT]  # convert the single subject to a list with one element
        elif CONFIG.EEG.MODE == 'CROSS_SUBJECT':
            """ Do cross-subject cross-fold validation for all subjects specified in data structure dataset """
            dataset = DATASETS[str(CONFIG.EEG.DS_NAME)]  # Initialize the read-only data structure 'dataset' which
            if CONFIG.EEG.CSV_SUBJECTS == 'All':
                # take all subjects defined in the dataset specific constant ALL_SUBJECTS
                CONFIG.EEG.CSV_SUBJECTS = dataset.CONSTANTS.ALL_SUBJECTS
                subjects = CONFIG.EEG.CSV_SUBJECTS
            else:
                # take the subjects specified in the corresponding yaml file
                subjects = CONFIG.EEG.CSV_SUBJECTS

            run_data = MLRunData(CONFIG.MI.SPLITS, CONFIG.MI.N_CLASS, CONFIG.MI.EPOCHS, CONFIG.MI.SPLITS)
           #res_folder, run_data= cross_fold_validation_2(subjects=subjects, save_results=True, save_all=True, tensorboard=True)
           run_optuna()