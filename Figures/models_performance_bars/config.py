"""

"""

fig_config = {
    1:
    # - Test only on low resolution
    # - Results for objects (Deit) & faces (apvit)
    # - Models pretrained with: low, high, mixed
        {
            'models': {'deit': ['deit_blur0_BS128', 'deit_blur8_BS128', 'deit_blur0-8_BS128'],
                       'apvit': ['RAF_blur8_pretrained_blur0', 'RAF_blur8', 'RAF_blur8_pretrained0-8']},
            'test_blurs': [8],
            'colors': [
                (.3, .655, 0.18),  # Green (high)
                (.95, .812, .933),  # Pink (low)
                (1.0, 1.0, 0.0)  # Yellow (mixed)
            ],
            'x_lbls': ['High', 'Low', 'Mixed'],
            'y_lim': {'deit': [20, 80], 'apvit': [60, 85]},
            'sep_fig_by': 'models'
        },

    2: {
        # - Test models on low/high resolution
        # - Only faces (apvit)
        # - Models pretrained with: low, mixed
        'models': ['apvit'],
        'test_blurs': {0: ['RAF_blur0_pretrained8', 'RAF_blur0_pretrained0-8'],
                       8: ['RAF_blur8', 'RAF_blur8_pretrained0-8']},
        'colors': [
            (.745, .68, .83),  # Purple (low)
            (.5, .788, 0.5)  # Green (mixed)
        ],
        'x_lbls': ['Low', 'Mixed'],
        'y_lim': {'deit': [70, 90], 'apvit': [70, 90]},
        'sep_fig_by': 'test_blurs'
        },

    3: {
        # - Test on low/high resolution
        # - Only faces (apvit)
        # - Models pretrained with: low, mixed
        # - Freezed models.
        'models': ['apvit'],
        'test_blurs': {0: ['RAF_blur0_pretrained8_freeze', 'RAF_blur0_pretrained0-8_freeze'],
                       8: ['RAF_blur8_freeze', 'RAF_blur8_pretrained0-8_freeze_fix_projs']},
        'colors': [
            (.745, .68, .83),  # Purple (low)
            (.5, .788, 0.5)  # Green (mixed)
        ],
        'x_lbls': ['Low', 'Mixed'],
        'y_lim': {'apvit': [40, 75]},
        'sep_fig_by': 'test_blurs'
        },

    4: {
        # - Test on low/high resolution
        # - Only faces (apvit
        # - Pretrained with: low, mixed + in high-res test: also model pretrained with high (narrow bar)
        # - Freezed models.
        'models': ['apvit'],
        'test_blurs': {0: ['RAF_blur0_pretrained8_freeze', 'RAF_blur0_pretrained0-8_freeze', 'RAF_blur0_freeze'],
                       8: ['RAF_blur8_freeze', 'RAF_blur8_pretrained0-8_freeze_fix_projs']},
        'colors': [
            (.745, .68, .83),  # Purple (low)
            (.5, .788, 0.5),  # Green (mixed)
            # (1, .553, 0.384),  # Orange (High)
            (1, .753, 0.525),  # Orange (High)
        ],
        'x_lbls': ['Low', 'Mixed', 'High'],
        'y_lim': {'apvit': [40, 85]},
        'sep_fig_by': 'test_blurs',
        'widths': [.8, .8, .2]
        },

    5: {
        # - Test on low/high resolution
        # - Only faces (apvit
        # - Pretrained with: low, mixed + in high-res test: also model pretrained with high (narrow bar)
        #       Add pretrained on 8-16 in low-res test.
        # - Freezed models.
        'models': ['apvit'],
        'test_blurs': {0: ['RAF_blur0_pretrained8_freeze', 'RAF_blur0_pretrained0-8_freeze', 'RAF_blur0_freeze'],
                       8: ['RAF_blur8_freeze', 'RAF_blur8_pretrained0-8_freeze_fix_projs', 'RAF_blur8_pretrained8-16_freeze']},
        'colors': [
            (.745, .68, .83),  # Purple (low)
            (.5, .788, 0.5),  # Green (mixed)
            # (1, .553, 0.384),  # Orange (High)
            (1, .753, 0.525),  # Orange (High)
        ],
        'x_lbls': ['Low', 'Mixed', 'High'],
        'y_lim': {'apvit': [40, 85]},
        'sep_fig_by': 'test_blurs',
        'widths': [.8, .8, .2]
        },
}
