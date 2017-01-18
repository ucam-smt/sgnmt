"""Implementation of beam search for predictors with multiple
tokenizations.
"""
from abc import abstractmethod
import copy
import heapq
import logging
import codecs
import pywrapfst as fst

from cam.sgnmt import utils
from cam.sgnmt.decoding.core import Decoder, PartialHypothesis
from cam.sgnmt.predictors.automata import EPS_ID


def is_key_complete(key):
    """Returns true if the key is complete. Complete keys are marked
    with a blank symbol at the end of the string. A complete key
    corresponds to a full word, incomplete keys cannot be mapped to
    word IDs.

    Args:
        key (string): The key

    Returns:
        bool. Return true if the last character in ``key`` is blank.
    """
    return key and key[-1] == ' '


class WordMapper(object):
    """This class is responsible for the mapping between keys and word
    IDs. The multiseg beam search can produce words which are not in 
    the original word map. This mapper adds these words to 
   ``utils.trg_wmap``.

    This class uses the GoF design pattern singleton.
    """

    singleton = None
    """Singleton instance. Access via ``get_singleton()``. """

    @staticmethod
    def get_singleton():
        """Get singleton instance of the word mapper. This method 
        implements lazy initialization.
        
        Returns:
            WordMapper. Singleton ``WordMapper`` instance.
        """
        if not WordMapper.singleton:
            WordMapper.singleton = WordMapper()
        return WordMapper.singleton

    def __init__(self):
        """Creates a new mapper instance and synchronizes it with
        ``utils.trg_wmap``.
        """
        self.max_word_id = 3
        self.wmap_len = 0
        self.key2id = {}
        self.synchronize()
        self.reserved_keys = {'<unk> ': utils.UNK_ID,
                              '<eps> ': utils.UNK_ID,
                              '<epsilon> ': utils.UNK_ID,
                              '<s> ': utils.GO_ID,
                              '</s> ': utils.EOS_ID}

    def synchronize(self):
        """Synchronizes the internal state of this mapper with
        ``utils.trg_wmap``. This includes updating the reverse lookup
        table and finding the lowest free word ID which can be assigned
        to new words. 
        """
        if self.wmap_len == len(utils.trg_wmap):
            return
        self.key2id = {}
        self.max_word_id = 3
        for word_id, key in utils.trg_wmap.iteritems():
            self.max_word_id = max(self.max_word_id, word_id)
            self.key2id["%s " % key] = word_id
        self.wmap_len = len(utils.trg_wmap)

    def get_word_id(self, key):
        """Finds a word ID for the given key. If no such key is in the
        current word map, create a new entry in ``utils.trg_wmap``.

        Args:
            key (string): key to look up

        Returns:
            int. Word ID corresponding to ``key``. Add new word ID if
            the key cannot be found in ``utils.trg_wmap`` 
        """
        if not key:
            return utils.UNK_ID
        if key in self.reserved_keys:
            return self.reserved_keys[key]
        self.synchronize()
        if key in self.key2id:
            return self.key2id[key]
        self.max_word_id += 1
        utils.trg_wmap[self.max_word_id] = key[:-1]
        self.key2id[key] = self.max_word_id
        self.wmap_len += 1
        return self.max_word_id


class Tokenizer(object):
    """A tokenizer translates between token sequences and string keys.
    It is mainly responsible for matching token sequences from
    different predictors together.
    """
    
    @abstractmethod
    def tokens2key(self, tokens):
        """Convert a token sequence to a string key.
        
        Args:
            tokens (list): List of token IDs
        
        Returns:
            String. The key for the token sequence
        """
        raise NotImplementedError
    
    @abstractmethod
    def key2tokens(self, key):
        """Convert a key to a sequence of tokens. If this mapping is
        ambiguous, return one of the shortest mappings. Use UNK to
        match any (sub)string without token correspondence.
        
        Args:
            key (string): key to look up
        
        Returns:
            list. List of token IDs
        """
        raise NotImplementedError

    @abstractmethod
    def is_word_begin_token(self, token):
        """Returns true if ``token`` is only allowed at word begins. """
        raise NotImplementedError


class WordTokenizer(Tokenizer):
    """This tokenizer implements a purly word-level tokenization.
    Keys are generated according a standard word map.
    """
    
    def __init__(self, path):
        self.id2key = {}
        self.key2id = {}
        try:
            split = path.split(":", 1)
            max_id = int(split[0])
            path = split[1]
        except:
            max_id = utils.INF
        with codecs.open(path, encoding='utf-8') as f:
            for line in f:
                entry = line.strip().split()
                key = entry[0]
                word_id = int(entry[-1])
                if word_id < max_id and word_id != utils.UNK_ID:
                    self.id2key[word_id] = "%s " % key
                    self.key2id["%s " % key] = word_id
    
    def key2tokens(self, key):
        return [self.key2id.get(key, utils.UNK_ID)]
    
    def tokens2key(self, tokens):
        if len(tokens) != 1:
            return ""
        return self.id2key.get(tokens[0], "")

    def is_word_begin_token(self, token):
        return True


class EOWTokenizer(Tokenizer):
    """This tokenizer reads word maps with explicit </w> endings. This
    can be used for subword unit based tokenizers.
    """
    
    def __init__(self, path):
        self.id2key = {}
        self.key2id = {}
        with codecs.open(path, encoding='utf-8') as f:
            for line in f:
                key, word_id = line.strip().split()
                if word_id == str(utils.UNK_ID):
                    continue
                if key[-4:] == "</w>": 
                    key = "%s " % key[:-4]
                elif key in ['<s>', '</s>']:
                    key = "%s " % key
                self.id2key[int(word_id)] = key
                self.key2id[key] = int(word_id)
    
    def key2tokens(self, key):
        tokens = self._key2tokens_recursive(key)
        return tokens if tokens else [utils.UNK_ID]

    def _key2tokens_recursive(self, key, max_len = 100):
        if not key:
            return []
        if max_len <= 0:
            return None
        if key in self.key2id: # Match of the full key
            return [self.key2id[key]]
        if max_len <= 1:
            return None
        best_tokens = None
        for l in xrange(len(key)-1, 0, -1):
            if key[:l] in self.key2id:
                rest = self._key2tokens_recursive(key[l:], max_len-1)
                if not rest is None and len(rest) < max_len:
                    best_tokens = [self.key2id[key[:l]]] + rest
                    max_len = len(best_tokens) - 1
        return best_tokens
    
    def tokens2key(self, tokens):
        return ''.join([self.id2key.get(t, "") for t in tokens])

    def is_word_begin_token(self, token):
        return token in [utils.GO_ID, utils.EOS_ID]


class MixedTokenizer(Tokenizer):
    """This tokenizer allows to mix word- and character-level
    tokenizations like proposed by Wu et al. (2016). Words with
    <b>, <m>, and <e> prefixes are treated as character-level
    tokens, all others are completed word-level tokens
    """
    
    def __init__(self, path):
        self.word_key2id = {}
        self.b_key2id = {}
        self.m_key2id = {}
        self.e_key2id = {}
        self.id2key = {}
        self.mid_tokens = {}
        try:
            split = path.split(":", 1)
            max_id = int(split[0])
            path = split[1]
        except:
            max_id = utils.INF
        with codecs.open(path, encoding='utf-8') as f:
            for line in f:
                key, token_id_ = line.strip().split()
                token_id = int(token_id_)
                if token_id == utils.UNK_ID or token_id >= max_id:
                    continue
                if key[:3] == "<b>":
                    key = key[3:]
                    self.b_key2id[key] = token_id
                elif key[:3] == "<m>":
                    key = key[3:]
                    self.m_key2id[key] = token_id
                    self.mid_tokens[token_id] = True
                elif key[:3] == "<e>":
                    key = "%s " % key[3:]
                    self.e_key2id[key[:-1]] = token_id
                    self.mid_tokens[token_id] = True
                else:
                    key = "%s " % key
                    self.word_key2id[key] = token_id
                self.id2key[token_id] = key 
    
    def key2tokens(self, key):
        if not key:
            return []
        if key in self.word_key2id:
            return [self.word_key2id[key]]
        maps = [self.m_key2id] * len(key)
        if is_key_complete(key):
            maps = maps[:-2] + [self.e_key2id]
        maps[0] = self.b_key2id
        return [maps[idx].get(key[idx], utils.UNK_ID)
                    for idx in xrange(len(maps))]
    
    def tokens2key(self, tokens):
        return ''.join([self.id2key.get(t, "") for t in tokens])

    def is_word_begin_token(self, token):
        return not self.mid_tokens.get(token, False)


class FSTTokenizer(Tokenizer):
    """This tokenizer reads in an FST which transduces a sequence
    of subword units to a sequence of characters which constitute
    the key. The characters must used the global target cmap.
    """
    
    EPS_ID = 0
    """OpenFST's reserved ID for epsilon arcs. """
    
    def __init__(self, path):
        """Loads subword->char FST, determinizes and minimizes it.
        
        Args:
            path (string): Path to an FST from subword unit to char
                           sequence
        """
        self.token2char_fst = utils.load_fst(path)
        self.token2char_fst.rmepsilon()
        self.token2char_fst.determinize()
        self.token2char_fst.minimize()
        self.word_begin_tokens = {arc.ilabel: True 
            for arc in self.token2char_fst.arcs(self.token2char_fst.start())}
        self.char2token_fst = fst.Fst(self.token2char_fst)
        self.char2token_fst.invert()
        self.cmap = dict(utils.trg_cmap)
        self.cmap[" "] = self.cmap["</w>"]
        del self.cmap["</w>"]
        self.inv_cmap = {(i,c) for c,i in self.cmap.iteritems()}
    
    def key2tokens(self, key):
        idxs = [self.cmap.get(c, utils.UNK_ID) for c in key]
        tokens = self._transduce(self.char2token_fst, idxs)
        return tokens if tokens else [utils.UNK_ID]
    
    def tokens2key(self, tokens):
        idxs = self._transduce(self.token2char_fst, tokens)
        if not idxs:
            return ""
        return ''.join([self.inv_cmap.get(i, '') for i in idxs])
    
    def _transduce(self, trans_fst, seq):
        """Returns the output sequence produced by ``trans_fst`` when
        consuming ``seq``. We don't check if the last state is a final
        state
        """
        return self._dfs(trans_fst, trans_fst.start(), seq, [])
            
    def _dfs(self, trans_fst, root, in_seq, out_seq_stub):
        """Perform DFS as subroutine of ``_transduce`` """
        if not in_seq:
            return out_seq_stub
        for arc in trans_fst.arcs(root):
            out_seq = None
            if arc.ilabel == EPS_ID:
                out_seq = self._dfs(trans_fst,
                                    arc.nextstate,
                                    in_seq, 
                                    out_seq_stub)
            elif arc.ilabel == in_seq[0]:
                out_seq = self._dfs(trans_fst, 
                                    arc.nextstate, 
                                    in_seq[1:], 
                                    out_seq_stub + [arc.olabel])
            if out_seq:
                return out_seq
        return None

    def is_word_begin_token(self, token):
        """Returns true if there is an arc labeled with ``token`` from
        the start state in the token2char FST.
        
        Args:
            token (int): token ID
        
        Returns:
            bool. True if a word can start with ``token`` 
        """
        return token in self.word_begin_tokens


class PredictorStub(object):
    """A predictor stub models the state of a predictor given a
    continuation.
    """
    
    def __init__(self, tokens, pred_state):
        """Creates a new stub for a certain predictor.
        
        Args:
            tokens (list): List of token IDs which correspond to the
                           key
            pred_state (object): Predictor state before consuming
                                 the last token in ``tokens``
        """
        self.tokens = tokens
        self.pred_state = pred_state
        self.score = 0.0
        self.score_pos = 0
    
    def has_full_score(self):
        """Returns true if the full token sequence has been scored with
        the predictor, i.e. ``self.score`` is the final predictor
        score.
        """
        return self.score_pos == len(self.tokens)

    def score_next(self, token_score):
        """Can be called when the continuation is expanded and the 
        score of the next token is available
        
        Args:
            token_score (float): Predictor score of 
                                 self.tokens[self.score_pos]
        """
        self.score += token_score
        self.score_pos += 1
    
    def expand(self, token, token_score, pred_state):
        """Creates a new predictor stub by adding a (scored) token.
        
        Args:
            token (int): Token ID to add
            token_score (float): Token score of the added token
            pred_state (object): predictor state before consuming
                                 the added token
        """
        new_stub = PredictorStub(self.tokens + [token], pred_state)
        new_stub.score_pos = self.score_pos + 1
        new_stub.score = self.score + token_score
        return new_stub


class Continuation(object):
    """A continuation is a partial hypothesis plus the next word. A
    continuation can be incomplete if predictors use finer grained
    tokenization and the score is not final yet.
    """
    
    def __init__(self, parent_hypo, pred_stubs, key = ''):
        """Create a new continuation.
        
        Args:
            parent_hypo (PartialHypothesis): hypo object encoding the
                                             state at the last word
                                             boundary
            pred_stubs (list): List of ``PredictorStub`` objects, one
                               for each predictor
            key (string): The lead key for this continuation. All stubs
                          must be consistent with this key
        """
        self.parent_hypo = parent_hypo
        self.pred_stubs = pred_stubs
        self.key = key
        self.score = 0.0
    
    def is_complete(self):
        """Returns true if all predictor stubs are completed, i.e. 
        the continuation can be mapped unambiguously to a word and the
        score is final.
        """
        return all([s and s.has_full_score() for s in self.pred_stubs])
    
    def calculate_score(self, pred_weights, defaults = []):
        """Calculates the full word score for this continuation using 
        the predictor stub scores.
        
        Args:
            pred_weights (list): Predictor weights. Length of this list
                                 must match the number of stubs
            defaults (list): Score which should be used if a predictor
                             stub is set to None
        
        Returns:
            float. Full score of this continuation, or an optimistic
            estimate if the continuation is not complete.
        """
        return sum(map(lambda x: x[0]*x[1],
                       zip(pred_weights,
                           [s.score if s else defaults[pidx]
                                  for pidx, s in enumerate(self.pred_stubs)])))
        
    def generate_expanded_hypo(self, decoder):
        """This can be used to create a new ``PartialHypothesis`` which
        reflects the state after this continuation. This involves
        expanding the history by ``word``, updating score and score_
        breakdown, and consuming the last tokens in the stub to save
        the final predictor states. If the continuation is complete, 
        this will result in a new word level hypothesis. If not, the
        generated hypo will indicate an incomplete word at the last
        position by using the word ID -1.
        """
        score_breakdown = []
        pred_weights = []
        for idx,(p, w) in enumerate(decoder.predictors):
            p.set_state(copy.deepcopy(self.pred_stubs[idx].pred_state))
            p.consume(self.pred_stubs[idx].tokens[-1])
            score_breakdown.append((self.pred_stubs[idx].score, w))
            pred_weights.append(w)
        word_id = WordMapper.get_singleton().get_word_id(self.key)
        return self.parent_hypo.expand(word_id,
                                       decoder.get_predictor_states(),
                                       self.calculate_score(pred_weights),
                                       score_breakdown)
    
    def expand(self, decoder):
        for pidx,(p, _) in enumerate(decoder.predictors):
            stub = self.pred_stubs[pidx]
            if not stub.has_full_score():
                p.set_state(copy.deepcopy(stub.pred_state))
                p.consume(stub.tokens[stub.score_pos-1])
                posterior = p.predict_next()
                stub.score_next(utils.common_get(
                                             posterior,
                                             stub.tokens[stub.score_pos],
                                             p.get_unk_probability(posterior)))
                stub.pred_state = p.get_state()


class MultisegBeamDecoder(Decoder):
    """This is a version of beam search which can handle predictors 
    with differing tokenizations. We assume that all tokenizations are
    consistent with words, i.e. no token crosses word boundaries. The
    search simulates beam search on the word level. At each time step, 
    we keep the n best hypotheses on the word level. Predictor scores
    on finer-grained tokens are collapsed into a single score s.t. they
    can be combined with scores from other predictors. This decoder can 
    produce words without entry in the word map. In this case, words 
    are added to ``utils.trg_wmap``. Consider using the ``output_chars``
    option to avoid dealing with the updated word map in the output.
    """
    
    def __init__(self,
                 decoder_args,
                 hypo_recombination,
                 beam_size,
                 tokenizations,
                 early_stopping = True,
                 max_word_len = 25):
        """Creates a new beam decoder instance for predictors with
        multiple tokenizations.
        
        Args:
            decoder_args (object): Decoder configuration passed through
                                   from the configuration API.
            hypo_recombination (bool): Activates hypo recombination 
            beam_size (int): Absolute beam size. A beam of 12 means
                             that we keep track of 12 active hypothesis
            tokenizations (string): Comma separated strings describing 
                                    the predictor tokenizations
            early_stopping (bool): If true, we stop when the best
                                   scoring hypothesis ends with </S>.
                                   If false, we stop when all hypotheses
                                   end with </S>. Enable if you are
                                   only interested in the single best
                                   decoding result. If you want to 
                                   create full 12-best lists, disable
            max_word_len (int): Maximum length of a single word
        """
        super(MultisegBeamDecoder, self).__init__(decoder_args)
        self.hypo_recombination = hypo_recombination
        self.beam_size = beam_size
        self.stop_criterion = self._best_eos if early_stopping else self._all_eos
        self.toks = []
        self.max_word_len = max_word_len
        if not tokenizations:
            logging.fatal("Specify --multiseg_tokenizations!")
        for tok_config in tokenizations.split(","):
            if tok_config[:6] == "mixed:":
                tok = MixedTokenizer(tok_config[6:])
            elif tok_config[:4] == "eow:":
                tok = EOWTokenizer(tok_config[4:])
            else:
                if tok_config[:5] == "word:":
                    tok_config = tok_config[5:]
                tok = WordTokenizer(tok_config)
            self.toks.append(tok)

    def _best_eos(self, hypos):
        """Returns true if the best hypothesis ends with </S>"""
        return hypos[0].get_last_word() != utils.EOS_ID

    def _all_eos(self, hypos):
        """Returns true if the all hypotheses end with </S>"""
        for hypo in hypos[:self.beam_size]:
            if hypo.get_last_word() != utils.EOS_ID:
                return True
        return False
    
    def _rebuild_hypo_list(self, hypos, new_hypo):
        """Add new_hypo to the list of n best complete hypos.
        Implements hypothesis recombination.
        
        Returns:
            list. Sorted list of n best hypos in hypos + new_hypo
        """
        if not self.hypo_recombination:
            hypos.append(new_hypo)
        else:
            combined = False
            for idx,hypo in list(enumerate(hypos)):
                if hypo.predictor_states and self.are_equal_predictor_states(
                                                    hypo.predictor_states,
                                                    new_hypo.predictor_states):
                    if hypo.score >= new_hypo.score: # Keep old one
                        hypo1 = hypo
                        hypo2 = new_hypo
                    else: # Discard old one
                        hypo1 = new_hypo
                        hypo2 = hypo
                        hypos[idx] = new_hypo
                    logging.debug("Hypo recombination: %s > %s" % (
                        hypo1.trgt_sentence, hypo2.trgt_sentence))
                    combined = True
                    break
            if not combined:
                hypos.append(new_hypo)
        hypos.sort(key=lambda h: h.score, reverse=True)
        return hypos[:self.beam_size]
    
    def _get_word_initial_posteriors(self, hypo):
        """Call ``predict_next`` on all predictors to get the 
        distributions over the first tokens of the next word.

        Args:
            hypo (PartialHypothesis): Defines the predictor states 

        Returns:
            list. List of posterior vectors for each predictor. The
            UNK scores are added to the vectors.
        """
        self.apply_predictors_count += 1
        self.set_predictor_states(hypo.predictor_states)
        posteriors = []
        for p, _ in self.predictors:
            posterior = p.predict_next()
            posterior[utils.UNK_ID] = p.get_unk_probability(posterior) 
            posteriors.append(posterior)
        return posteriors

    def _get_initial_stubs(self, predictor, start_posterior, min_score):
        """Get the initial predictor stubs for full word search with a 
        single predictor. 
        """
        stubs = []
        pred_state = predictor.get_state()
        for t, s in utils.common_iterable(start_posterior):
            stub = PredictorStub([t], pred_state)
            stub.score_next(s)
            if stub.score >= min_score:
                stubs.append(stub)
        stubs.sort(key=lambda s: s.score, reverse=True)
        return stubs
   
    def _best_keys_complete(self, stubs, tok):
        """Stopping criterion for single predictor full word search.
        We stop full word search if the n best stubs are complete.
        """
        return all([is_key_complete(tok.tokens2key(s.tokens)) 
                                           for s in stubs[:self.beam_size]])

    def _search_full_words(self, predictor, start_posterior, tok, min_score):
        stubs = self._get_initial_stubs(predictor, start_posterior, min_score)
        while not self._best_keys_complete(stubs, tok):
            next_stubs = []
            for stub in stubs[:self.beam_size]:
                key = tok.tokens2key(stub.tokens)
                if (not key) or len(key) > self.max_word_len:
                    continue
                if is_key_complete(key):
                    next_stubs.append(stub)
                    continue
                predictor.set_state(copy.deepcopy(stub.pred_state))
                predictor.consume(stub.tokens[-1])
                posterior = predictor.predict_next()
                pred_state = predictor.get_state()
                for t, s in utils.common_iterable(posterior):
                    if t != utils.UNK_ID and not tok.is_word_begin_token(t):
                        child_stub = stub.expand(t, s, pred_state)
                        if child_stub.score >= min_score:
                            next_stubs.append(child_stub)
            stubs = next_stubs
            stubs.sort(key=lambda s: s.score, reverse=True)
        return stubs
    
    def _get_complete_continuations(self, hypo, min_hypo_score):
        """This is a generator which yields the complete continuations 
        of ``hypo`` in descending order of score
        """
        min_score = min_hypo_score - hypo.score
        if min_score > 0.0:
            return
        
        pred_weights = map(lambda el: el[1], self.predictors)
        # Get initial continuations by searching with predictors separately
        start_posteriors = self._get_word_initial_posteriors(hypo)
        pred_states = self.get_predictor_states()
        keys = {}
        for pidx, (p,w) in enumerate(self.predictors):
            stubs = self._search_full_words(p,
                                            start_posteriors[pidx],
                                            self.toks[pidx],
                                            min_score / w)
            n_added = 0
            for stub in stubs:
                key = self.toks[pidx].tokens2key(stub.tokens)
                if is_key_complete(key):
                    if key in keys: # Add to existing continuation
                        prev_stub = keys[key].pred_stubs[pidx]
                        if prev_stub is None or prev_stub.score < stub.score:
                            keys[key].pred_stubs[pidx] = stub
                    elif n_added < self.beam_size: # Create new continuation
                        n_added += 1
                        stubs = [None] * len(self.predictors)
                        stubs[pidx] = stub
                        keys[key] = Continuation(hypo, stubs, key)
        # Fill in stubs which are set to None
        for cont in keys.itervalues():
            for pidx in xrange(len(self.predictors)):
                if cont.pred_stubs[pidx] is None:
                    stub = PredictorStub(self.toks[pidx].key2tokens(cont.key),
                                         pred_states[pidx])
                    stub.score_next(utils.common_get(
                                         start_posteriors[pidx],
                                         stub.tokens[0],
                                         start_posteriors[pidx][utils.UNK_ID]))
                    cont.pred_stubs[pidx] = stub
        conts = [(-c.calculate_score(pred_weights), c) for c in keys.itervalues()]
        heapq.heapify(conts)
        # Iterate through conts, expand if necessary, yield if complete
        while conts:
            s,cont = heapq.heappop(conts)
            if cont.is_complete():
                yield -s,cont
            else: # Need to rescore with sec predictors
                cont.expand(self)
                heapq.heappush(conts, (-cont.calculate_score(pred_weights), cont))
    
    def decode(self, src_sentence):
        """Decodes a single source sentence using beam search. """
        self.initialize_predictors(src_sentence)
        hypos = [PartialHypothesis(self.get_predictor_states())]
        guard_hypo = PartialHypothesis(None)
        guard_hypo.score = utils.NEG_INF
        it = 0
        while self.stop_criterion(hypos):
            if it > self.max_len: # prevent infinite loops
                break
            it = it + 1
            next_hypos = [guard_hypo]
            #print("HYPOS")
            #for hypo in hypos:
            #    print("it%d: %s (%f)" % (it, utils.apply_trg_wmap(hypo.trgt_sentence), hypo.score))
            for hypo in hypos:
                #print("H: %s (%f: %f, %f, %f)" % (utils.apply_trg_wmap(hypo.trgt_sentence), hypo.score, sum([s[0][0] for s in hypo.score_breakdown]), sum([s[1][0] for s in hypo.score_breakdown]), sum([s[2][0] for s in hypo.score_breakdown])))
                if hypo.get_last_word() == utils.EOS_ID:
                    next_hypos = self._rebuild_hypo_list(next_hypos, hypo)
                for s, cont in self._get_complete_continuations(
                                                        hypo,
                                                        next_hypos[-1].score):
                    if hypo.score + s < next_hypos[-1].score:
                        break
                    next_hypos = self._rebuild_hypo_list(
                                            next_hypos,
                                            cont.generate_expanded_hypo(self))
            hypos = [h for h in next_hypos if h.score > utils.NEG_INF]  
        for hypo in hypos:
            if hypo.get_last_word() == utils.EOS_ID:
                self.add_full_hypo(hypo.generate_full_hypothesis()) 
        if not self.full_hypos:
            logging.warn("No complete hypotheses found for %s" % src_sentence)
            for hypo in hypos:
                self.add_full_hypo(hypo.generate_full_hypothesis())
        return self.get_full_hypos_sorted()
