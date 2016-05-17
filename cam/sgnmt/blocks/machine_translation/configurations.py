def get_config_cs2en():
    config = {}

    # Model related -----------------------------------------------------------

    # Sequences longer than this will be discarded
    config['seq_len'] = 50

    # Number of hidden units in encoder/decoder GRU
    config['enc_nhids'] = 1000
    config['dec_nhids'] = 1000

    # Dimension of the word embedding matrix in encoder/decoder
    config['enc_embed'] = 620
    config['dec_embed'] = 620

    # Where to save model, this corresponds to 'prefix' in groundhog
    config['saveto'] = 'search_model_cs2en'

    # Optimization related ----------------------------------------------------

    # Batch size
    config['batch_size'] = 80

    # This many batches will be read ahead and sorted
    config['sort_k_batches'] = 12

    # Optimization step rule
    config['step_rule'] = 'AdaDelta'

    # Gradient clipping threshold
    config['step_clipping'] = 1.

    # Std of weight initialization
    config['weight_scale'] = 0.01

    # Regularization related --------------------------------------------------

    # Weight noise flag for feed forward layers
    config['weight_noise_ff'] = False

    # Weight noise flag for recurrent layers
    config['weight_noise_rec'] = False

    # Dropout ratio, applied only after readout maxout
    config['dropout'] = 1.0

    # Vocabulary/dataset related ----------------------------------------------

    # Root directory for dataset
    datadir = './data/'

    # Module name of the stream that will be used
    config['stream'] = 'stream'

    # Source and target vocabularies
    config['src_vocab'] = datadir + 'vocab.cs-en.cs.pkl'
    config['trg_vocab'] = datadir + 'vocab.cs-en.en.pkl'

    # Source and target datasets
    config['src_data'] = datadir + 'news-commentary-v10.cs-en.cs.tok.shuf'
    config['trg_data'] = datadir + 'news-commentary-v10.cs-en.en.tok.shuf'

    # Source and target vocabulary sizes, should include bos, eos, unk tokens
    config['src_vocab_size'] = 30000
    config['trg_vocab_size'] = 30000

    # Special tokens and indexes
    config['unk_id'] = 1
    config['bos_token'] = '<S>'
    config['eos_token'] = '</S>'
    config['unk_token'] = '<UNK>'

    # Early stopping based on bleu related ------------------------------------

    # Normalize cost according to sequence length after beam-search
    config['normalized_bleu'] = True

    # Bleu script that will be used (moses multi-perl in this case)
    config['bleu_script'] = datadir + 'multi-bleu.perl'

    # Validation set source file
    config['val_set'] = datadir + 'newstest2013.cs.tok'

    # Validation set gold file
    config['val_set_grndtruth'] = datadir + 'newstest2013.en.tok'

    # Print validation output to file
    config['output_val_set'] = True

    # Validation output file
    config['val_set_out'] = config['saveto'] + '/validation_out.txt'

    # Beam-size
    config['beam_size'] = 12

    # Timing/monitoring related -----------------------------------------------

    # Maximum number of updates
    config['finish_after'] = 1000000

    # Reload model from files if exist
    config['reload'] = True

    # Save model after this many updates
    config['save_freq'] = 500

    # Show samples from model after this many updates
    config['sampling_freq'] = 13

    # Show this many samples at each sampling
    config['hook_samples'] = 2

    # Validate bleu after this many updates
    config['bleu_val_freq'] = 5000

    # Start bleu validation after this many updates
    config['val_burn_in'] = 80000

    return config

def get_config_gnmt():
    config = {}

    # Model related -----------------------------------------------------------

    # Sequences longer than this will be discarded
    config['seq_len'] = 50

    # Number of hidden units in encoder/decoder GRU
    config['enc_nhids'] = 1000
    config['dec_nhids'] = 1000

    # Dimension of the word embedding matrix in encoder/decoder
    config['enc_embed'] = 620
    config['dec_embed'] = 620

    # Where to save model, this corresponds to 'prefix' in groundhog
    config['saveto'] = './train'

    # Optimization related ----------------------------------------------------

    # Batch size
    config['batch_size'] = 80

    # This many batches will be read ahead and sorted
    config['sort_k_batches'] = 12

    # Optimization step rule
    config['step_rule'] = 'AdaDelta'

    # Gradient clipping threshold
    config['step_clipping'] = 1.

    # Std of weight initialization
    config['weight_scale'] = 0.01

    # Regularization related --------------------------------------------------

    # Weight noise flag for feed forward layers
    config['weight_noise_ff'] = 0.0

    # Weight noise flag for recurrent layers
    config['weight_noise_rec'] = False

    # Dropout ratio, applied only after readout maxout
    config['dropout'] = 1.0

    # Vocabulary/dataset related ----------------------------------------------

    # Root directory for dataset
    datadir = './data/'
    scriptsdir = '../scripts/'

    # Module name of the stream that will be used
    config['stream'] = 'stream'

    # Source and target vocabularies
    #config['src_vocab'] = datadir + 'vocab.cs-en.cs.pkl'
    #config['trg_vocab'] = datadir + 'vocab.cs-en.en.pkl'

    # Source and target datasets
    config['src_data'] = datadir + 'train.ids.shuf.en'
    config['trg_data'] = datadir + 'train.ids.shuf.fr'

    # Source and target vocabulary sizes, should include bos, eos, unk tokens
    config['src_vocab_size'] = 30003
    config['trg_vocab_size'] = 30003

    # Special tokens and indexes
    config['unk_id'] = 0
    config['bos_token'] = '_GO'
    config['eos_token'] = '_EOS'
    config['unk_token'] = '_UNK'

    # Early stopping based on bleu related ------------------------------------

    # Normalize cost according to sequence length after beam-search
    config['normalized_bleu'] = True

    # Bleu script that will be used (moses multi-perl in this case)
    config['bleu_script'] = 'perl ' + scriptsdir + 'multi-bleu.perl %s <'

    # Validation set source file
    config['val_set'] = datadir + 'dev.ids.en'

    # Validation set gold file
    config['val_set_grndtruth'] = datadir + 'dev.ids.fr'

    # Print validation output to file
    config['output_val_set'] = True

    # Validation output file
    config['val_set_out'] = config['saveto'] + '/validation_out.txt'

    # Beam-size
    config['beam_size'] = 12

    # Timing/monitoring related -----------------------------------------------

    # Maximum number of updates
    config['finish_after'] = 1000000

    # Reload model from files if exist
    config['reload'] = True

    # Save model after this many updates
    config['save_freq'] = 500

    # Show samples from model after this many updates
    config['sampling_freq'] = 13

    # Show this many samples at each sampling
    config['hook_samples'] = 2

    # Validate bleu after this many updates
    config['bleu_val_freq'] = 5000

    # Start bleu validation after this many updates
    config['val_burn_in'] = 80000

    # fs439: Blocks originally creates dumps of the entire main loop
    # when the BLEU on the dev set improves. This, however, cannot be
    # read to load parameters from, so we create BEST_BLEU_PARAMS*
    # files instead. Set the following parameter to true if you still
    # want to create the old style archives
    config['store_full_main_loop'] = False
    
    # fs439: Fix embeddings when training
    config['fix_embeddings'] = False

    return config
