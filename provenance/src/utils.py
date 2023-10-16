import os
def validate_args(dataset,
                  descriptor_type,
                  matching_strategy,
                  alignment_strategy,
                  top_k,
                  min_keypoints,
                  max_queuesize,
                  use_area,
                  min_area=None,
                  same_class=False,
                  visualize=False):

    # Assert dataset is string and exists
    assert type(dataset) is str, "Dataset argument must be string"
    assert os.path.isfile(dataset) , f"Dataset {dataset} does not exists"

    # Assert descriptor type is implemented
    implemented_descriptors = [
        'sila',
        'cv_rsift_heq',
        'cv_rsift',
        'cv_sift',
        'cv_sift_heq',
        'vlfeat_phow',
        'vlfeat_dsift',
        'vlfeat_rsift',
        'vlfeat_rsift_heq',
        'vlfeat_sift',
        'vlfeat_sift_heq',
    ]
    assert descriptor_type in implemented_descriptors, f"Descriptor {descriptor_type} not implemented"

    # Assert matching strategy exists
    implemented_matching = [
        'BF',
        'FLANN'
    ]
    assert matching_strategy in implemented_matching, f"Matching Strategy {matching_strategy} not implemented"

    # Assert alignment strategy exist
    implemented_alignments= [
        'cluster',
        'CV_RANSAC',
        "CV_LMEDS",
        "CV_RHO",
        'CV_DEGENSAC',
        'CV_MAGSAC'
    ]
    assert alignment_strategy in implemented_alignments, f"Alignment Strategy {matching_strategy} not implemented"

    # Assert that if the area is used, min_area is not None
    if use_area:
        assert min_area, "Argument 'use_area' set, but min_area not set"

    return dataset, descriptor_type, matching_strategy, alignment_strategy,\
          top_k, min_keypoints, max_queuesize, min_area, same_class, visualize
    

