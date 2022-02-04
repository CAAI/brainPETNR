import sys
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
import torch
import torchio as tio
from torch.utils.data import DataLoader
from rhtorch.utilities.config import UserConfig
from rhtorch.utilities.modules import recursive_find_python_class
from rhtorch.utilities.modules import find_best_checkpoint


def infer_data_from_model(model,
                          subject,
                          patch_size=None,
                          patch_overlap=None,
                          batch_size=1,
                          gpu=True):
    """Infer a full volume given a trained model for 1 patient

    Args:
        model (torch.nn.Module): trained pytorch model
        subject (torchio.Subject): Subject instance from TorchIO library
        patch_size (list, optional): Patch size (from config). Defaults to None.
        patch_overlap (int or list, optional): Patch overlap. Defaults to None.
        batch_size (int, optional): batch_size (from_config). Defaults to 1.

    Returns:
        [np.ndarray]: Full volume inferred from model
    """

    grid_sampler = tio.data.GridSampler(subject, patch_size, patch_overlap)
    patch_loader = DataLoader(grid_sampler, batch_size=batch_size)
    aggregator = tio.data.GridAggregator(grid_sampler, overlap_mode='average')
    with torch.no_grad():
        for patches_batch in patch_loader:
            patch_x, _ = model.prepare_batch(patches_batch)
            if gpu:
                patch_x = patch_x.to('cuda')
            locations = patches_batch[tio.LOCATION]
            patch_y = model(patch_x)
            aggregator.add_batch(patch_y, locations)
    return aggregator.get_output_tensor()


def main():
    parser = argparse.ArgumentParser(
        description='Infer new data from input model.')
    parser.add_argument("-c",
                        "--config",
                        help="Config file of saved model",
                        type=str,
                        default='config.yaml')
    parser.add_argument("-im",
                        "--infer-mode",
                        help="Which patient set to use: test or eval.",
                        type=str,
                        default='test')
    parser.add_argument("--gpu",
                        help="Use GPU. Will only use CPU if false.",
                        action="store_true",
                        default=True)

    args = parser.parse_args()
    infer_mode = args.infer_mode

    # load configs in inference mode
    user_configs = UserConfig(args, mode='infer')
    model_dir = user_configs.rootdir
    configs = user_configs.hparams

    project_dir = Path(configs['project_dir'])
    model_name = configs['model_name']
    infer_dir_name = f'inferences_{infer_mode}_set'
    infer_dir = model_dir.joinpath(infer_dir_name)
    infer_dir.mkdir(parents=True, exist_ok=True)
    data_shape_in = configs['data_shape_in']
    patch_size = configs['patch_size']
    patch_overlap = int(np.min(patch_size) / 2)

    # load the test data
    sys.path.insert(1, str(project_dir))
    import data_generator
    data_gen = getattr(data_generator, configs['data_generator'])
    data_module = data_gen(configs, args.test)
    test_subjects = data_module.prepare_patient_data(infer_mode)
    test_set = tio.SubjectsDataset(test_subjects)

    # load the model
    module_name = recursive_find_python_class(configs['module'])
    model = module_name(configs, data_shape_in)
    # Load the final (best) model
    if 'best_model' in configs:
        ckpt_path = Path(configs['best_model'])
        epoch_suffix = ''
    # Not done training. Load the most recent (best) ckpt
    else:
        ckpt_path = find_best_checkpoint(
            project_dir.joinpath('trained_models', model_name, 'checkpoints'))
        epoch_suffix = None
    ckpt = torch.load(ckpt_path)

    if epoch_suffix is None:
        epoch_suffix = f"_e={ckpt['epoch']}"
    model.load_state_dict(ckpt['state_dict'])

    if args.gpu:
        model.cuda()
    model.eval()

    # for patient in tqdm(getattr(data_module, set_type)):
    for patient in tqdm(test_set):
        patient_id = patient.id
        out_subdir = infer_dir.joinpath(patient_id)
        out_subdir.mkdir(parents=True, exist_ok=True)
        output_file = out_subdir.joinpath(
            f'Inferred_{model_name}{epoch_suffix}.nii.gz')

        # check if recontruction already done
        if not output_file.exists():
            # full volume inference - returns np.ndarray
            full_volume = infer_data_from_model(model,
                                                patient,
                                                patch_size,
                                                patch_overlap,
                                                gpu=args.gpu)

            # rescale -- need a better way to revert preprocessing steps from DataModule
            full_volume = full_volume * configs['pet_normalization_constant']
            final_image = tio.ScalarImage(tensor=full_volume,
                                          affine=patient.input0.affine)

            # pad if there was cropping
            if 'cropping' in configs:
                crop_config = configs['cropping']
                pad = tio.Pad((*crop_config['x_lim'], *crop_config['y_lim'],
                               *crop_config['z_lim']))
                final_image = pad(final_image)

            # save nifty with torchio
            final_image.save(output_file)

        else:
            print(
                f'Data for {patient.id} already reconstructed with model {model_name}'
            )


if __name__ == '__main__':
    main()