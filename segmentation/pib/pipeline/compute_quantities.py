from nipype.interfaces.fsl import ImageStats


def _compute_median(region_path, pet_image_path):
    stats = ImageStats(in_file=pet_image_path,
                       op_string="-k %s -p 50",
                       mask_file=region_path)
    result = stats.run()
    median = result.outputs.out_stat
    return median


def _compute_volume(region_path):
    stats = ImageStats(in_file=region_path,
                       op_string="-V")
    result = stats.run()
    volume = result.outputs.out_stat
    return volume[-1]


def _compute_normalizing_activity(region_path, pet_image_path):
    stats = ImageStats(in_file=pet_image_path,
                       op_string="-k %s -p 50",
                       mask_file=region_path)
    result = stats.run()
    median = result.outputs.out_stat
    return median


def _compute_volume_weighted_sum(region_paths, pet_image_path, reference_region_path):
    normalizing_activity = _compute_normalizing_activity(
        reference_region_path, pet_image_path)

    median_activities = [_compute_median(region_path, pet_image_path)
                         for region_path in region_paths]

    volumes = [_compute_volume(region_path)
               for region_path in region_paths]

    volume_weighted_sum = (sum([activity * volume
                                for activity, volume in zip(median_activities,
                                                            volumes)])
                           / float(normalizing_activity))

    return volume_weighted_sum


def compute_normalizing_activity(region_path, pet_image_path):
    # from pipeline.compute_quantities import _compute_normalizing_activity
    from compute_quantities import _compute_normalizing_activity
    return _compute_normalizing_activity(region_path, pet_image_path)


def compute_volumes(region_paths):
    from compute_quantities import _compute_volume

    volumes = [_compute_volume(region_path)
               for region_path in region_paths]

    return volumes


def compute_suvrs(region_paths, pet_image_path, reference_region_path):
    from compute_quantities import _compute_normalizing_activity, _compute_median

    normalizing_activity = _compute_normalizing_activity(
        reference_region_path, pet_image_path)

    suvrs = [_compute_median(region_path, pet_image_path) / normalizing_activity
             for region_path in region_paths]

    return suvrs


def compute_volume_weighted_mean(region_paths, pet_image_path, reference_region_path):
    from compute_quantities import _compute_volume_weighted_sum, compute_volumes

    volume_weighted_sum = _compute_volume_weighted_sum(
        region_paths, pet_image_path, reference_region_path)

    volumes = compute_volumes(region_paths)

    volume_weighted_mean = volume_weighted_sum / sum(volumes)

    return volume_weighted_mean
