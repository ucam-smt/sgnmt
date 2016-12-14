"""Implementation of beam search for predictors with multiple
tokenizations.
"""
from abc import abstractmethod
import copy
import heapq
import logging

from cam.sgnmt import utils
from cam.sgnmt.decoding.core import Decoder, PartialHypothesis


def is_key_complete(key):
    return key and key[-1] == ' '


class WordMapper(object):
    """This class is responsible for the mapping between keys and word
    IDs. The multiseg beam search can produce words which are not in 
    the original word map. This mapper adds these words to 
   ``utils.trg_wmap``.
    """
    singleton = None

    @staticmethod
    def get_singleton():
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


class WordTokenizer(Tokenizer):
    """This tokenizer implements a purly word-level tokenization.
    Keys are generated according a standard word map.
    """
    
    def __init__(self, path):
        self.id2key = {}
        self.key2id = {}
        with open(path) as f:
            for line in f:
                key, word_id = line.strip().split()
                self.id2key[int(word_id)] = "%s " % key
                self.key2id["%s " % key] = int(word_id)
    
    def key2tokens(self, key):
        return [self.key2id.get(key, utils.UNK_ID)]
    
    def tokens2key(self, tokens):
        if len(tokens) != 1:
            return ""
        return self.id2key.get(tokens[0], "")


class EOWTokenizer(Tokenizer):
    """This tokenizer reads word maps with explicit </w> endings. This
    can be used for subword unit based tokenizers.
    """
    
    def __init__(self, path):
        self.id2key = {}
        self.key2id = {}
        with open(path) as f:
            for line in f:
                key, word_id = line.strip().split()
                if key[-4:] == "</w>":
                    key = "%s " % key[:-4]
                self.id2key[int(word_id)] = key
                self.key2id[key] = int(word_id)
    
    def key2tokens(self, key, max_len = 100):
        if not key:
            return []
        if max_len <= 0:
            return None
        t = self.key2id.get(key)
        if t: # Match of the full key
            return [t]
        if max_len <= 1:
            return None
        best_tokens = None
        for l in xrange(len(key)-1, 0, -1):
            t = self.key2id.get(key[:l])
            if t:
                rest = self.key2tokens(key, max_len-1)
                if not rest is None and len(rest) < max_len:
                    best_tokens = [t] + rest
                    max_len = len(best_tokens) - 1
        return best_tokens
    
    def tokens2key(self, tokens):
        return ''.join([self.id2key.get(t, "") for t in tokens])


class MixedTokenizer(Tokenizer):
    """This tokenizer allows to mix word- and character-level
    tokenizations like proposed by Wu et al. (2016). Words with
    <b>, <m>, and <e> postfixes are treated as character-level
    tokens, all others are completed word-level tokens
    """
    
    def __init__(self, path):
        self.word_key2id = {}
        self.b_key2id = {}
        self.m_key2id = {}
        self.e_key2id = {}
        self.id2key = {}
        with open(path) as f:
            for line in f:
                key, token_id = line.strip().split()
                if key[-3:] == "<b>":
                    key = key[:-3]
                    self.b_key2id[key] = int(token_id)
                elif key[-3:] == "<m>":
                    key = key[:-3]
                    self.m_key2id[key] = int(token_id)
                elif key[-3:] == "<e>":
                    key = "%s " % key[:-3]
                    self.e_key2id[key[:-1]] = int(token_id)
                else:
                    key = "%s " % key
                    self.word_key2id[key] = int(token_id)
                self.id2key[int(token_id)] = key 
    
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
        return self.parent_hypo.expand(
                              WordMapper.get_singleton().get_word_id(self.key),
                              decoder.get_predictor_states(),
                              self.calculate_score(pred_weights),
                              score_breakdown)
    
    def expand(self, decoder):
        for pidx,(p, _) in enumerate(decoder.predictors):
            stub = self.pred_stubs[pidx]
            if not stub.has_full_score():
                p.set_state(copy.deepcopy(stub.pred_state))
                p.consume(stub.tokens[self.score_pos-1])
                posterior = p.predict_next()
                stub.score_next(posterior[stub.tokens[self.score_pos]])
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
                 early_stopping = True):
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
        """
        super(MultisegBeamDecoder, self).__init__(decoder_args)
        self.hypo_recombination = hypo_recombination
        self.beam_size = beam_size
        self.stop_criterion = self._best_eos if early_stopping else self._all_eos
        self.toks = []
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
        return hypos[-1].get_last_word() != utils.EOS_ID

    def _all_eos(self, hypos):
        """Returns true if the all hypotheses end with </S>"""
        for hypo in hypos:
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
                if self.are_equal_predictor_states(hypo.predictor_states,
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
        self.apply_predictors_count += 1
        self.set_predictor_states(hypo.predictor_states)
        posteriors = []
        for p, _ in self.predictors:
            posterior = p.predict_next()
            posterior[utils.UNK_ID] = p.get_unk_probability(posterior) 
            posteriors.append(posterior)
        return posteriors

    def _get_initial_stubs(self, predictor, start_posterior, min_score):
        stubs = []
        pred_state = predictor.get_state()
        for t, s in utils.common_iterable(start_posterior):
            stub = PredictorStub([t], pred_state)
            stub.score_next(s)
            if stub.score >= min_score:
                stubs.append(stub)
        stubs.sort(key=lambda s: s.score, reverse=True)
        return stubs
    
    def _search_full_words(self, predictor, start_posterior, tok, min_score):
        stubs = self._get_initial_stubs(predictor, start_posterior, min_score)
        best_key = tok.tokens2key(stubs[0].tokens) if stubs else " "
        while not is_key_complete(best_key):
            print("stubs: %d" % len(stubs))
            next_stubs = []
            for stub in stubs[:self.beam_size]:
                if is_key_complete(tok.tokens2key(stub.tokens)):
                    next_stubs.append(stub)
                    continue
                predictor.set_state(copy.deepcopy(stub.pred_state))
                predictor.consume(stub.tokens[-1])
                posterior = predictor.predict_next()
                pred_state = predictor.get_stat()
                for t, s in utils.common_iterable(posterior):
                    child_stub = stub.expand(t, s, pred_state)
                    if child_stub.score >= min_score:
                        next_stubs.append(child_stub)
            stubs = next_stubs
            stubs.sort(key=lambda s: s.score, reverse=True)
            best_key = tok.tokens2key(stubs[0].tokens) if stubs else " "
        complete_stubs = {}
        for stub in stubs:
            key = tok.tokens2key(stub.tokens)
            if is_key_complete(key):
                complete_stubs[key] = stub
        return complete_stubs
    
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
            key2stub = self._search_full_words(p,
                                               start_posteriors[pidx],
                                               self.toks[pidx],
                                               min_score / w)
            for key, stub in key2stub.iteritems():
                if key in keys: # Add to existing continuation
                    keys[key].pred_stubs[pidx] = stub
                else: # Create new continuation
                    stubs = [None] * len(self.predictors)
                    stubs[pidx] = stub
                    keys[key] = Continuation(hypo, stubs, key)
        # Fill in stubs which are set to None
        for cont in keys.itervalues():
            for pidx in xrange(len(self.predictors)):
                if cont.pred_stubs[pidx] is None:
                    stub = PredictorStub(self.toks[pidx].key2tokens(cont.key),
                                         pred_states[pidx])
                    stub.score_next(start_posteriors[pidx][stub.tokens[0]])
                    cont.stubs[pidx] = stub
        conts = [(-c.calculate_score(pred_weights), c) for c in keys.itervalues()]
        heapq.heapify(conts)
        print("%d conts, min score %f" % (len(conts), min_score))
        # Iterate through conts, expand if necessary, yield if complete
        while conts:
            s,cont = heapq.heappop(conts)
            if cont.is_complete():
                yield -s,cont
            else: # Need to rescore with sec predictors
                cont.expand(self)
                heapq.heappush(conts, (-cont.calculate_score(pred_weights), c))
    
    def decode(self, src_sentence):
        """Decodes a single source sentence using beam search. """
        self.initialize_predictors(src_sentence)
        hypos = [PartialHypothesis(self.get_predictor_states())]
        guard_hypo = PartialHypothesis()
        guard_hypo.score = utils.NEG_INF
        it = 0
        while self.stop_criterion(hypos):
            print("IT %d (%d hypos)" % (it, len(hypos)))
            if it > self.max_len: # prevent infinite loops
                break
            it = it + 1
            next_hypos = [guard_hypo]
            for hypo in hypos:
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
