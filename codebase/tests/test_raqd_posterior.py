import numpy as np

from raqd_posterior import STYLES, genome_to_unit, propose_batch, sobol_genomes


RANGES = ((4.0, 8.0), (.2, .8))


def observations():
    rows = []
    genomes = sobol_genomes(4, 11)
    for recipe, genome in enumerate(genomes):
        style, x = genome_to_unit(genome)
        for repeat in range(3):
            rows.append({'id': f'{recipe}_{repeat}', 'replicate_group': f'r{recipe}',
                         'genome': genome, 'valid': True,
                         'constraints_satisfied': recipe == 0,
                         'realized_descriptors': [4 + 3*x[0] + .01*repeat,
                                                  .25 + .3*x[1] + .005*repeat],
                         'measured_design_volume_fraction': .48 + .02*x[2] + .002*repeat,
                         'compliance_J': .004 + .001*x[0]})
    return rows


def test_sobol_genomes_obey_domain():
    genomes = sobol_genomes(16, 7)
    assert {g['style'] for g in genomes} == set(STYLES)
    assert all(3 <= g['cfg'] <= 9 and 3 <= g['sp_cfg'] <= 7 and
               20 <= g['sp_guide_w_peak'] <= 80 for g in genomes)


def test_posterior_batch_is_reproducible_and_diverse():
    first, diagnostics = propose_batch(observations(), 3, 99, RANGES, pool_size=256)
    second, _ = propose_batch(observations(), 3, 99, RANGES, pool_size=256)
    assert first == second
    assert diagnostics['usable_observations'] == 12
    signatures = {(j['genome']['style'], *(round(v, 5) for k, v in j['genome'].items()
                   if k != 'style')) for j in first}
    assert len(signatures) == 3
