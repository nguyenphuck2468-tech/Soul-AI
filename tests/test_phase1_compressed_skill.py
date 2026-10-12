import numpy as np

from minicraft import MiniCraftEnv
from minicraft.compressed_skill import CompressedObservationSkill
from minicraft.distilled_skill import fit_decision_tree, observation_features


def test_compressed_skill_uses_mapped_fallback_on_disagreement():
    observation = MiniCraftEnv(seed=400).observe()
    features = observation_features(observation)
    tree, _ = fit_decision_tree(
        np.zeros((4, features.size), dtype=np.float32),
        np.full(4, 8, dtype=np.int64),
        min_samples_leaf=1,
    )
    skill = CompressedObservationSkill('T_wood', {
        'schema_version': 1,
        'task': 'T_wood',
        'tree': tree,
    })

    proposal = skill.propose(observation)

    assert proposal.source == 'mapped_fallback'
    assert proposal.candidate_action == proposal.fallback_action
    assert proposal.learned_action == 8
