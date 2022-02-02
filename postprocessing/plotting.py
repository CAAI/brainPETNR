import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rhscripts.plotting import _PETRainbowCMAP
from postutils import _MidPointNorm, lr_model
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.gridspec import GridSpec
plt.style.use('/homes/raphael/brainPETNR/notebooks/postprocess/plots.mplstyle')

# sns.set_theme(style="white")

palette = 'Set2'
colors = ['mediumaquamarine', 'lightsalmon']

def brain_slices_grid_pib(data, 
                        patient_id, 
                        file_out, 
                        segment_data=None,
                        slices='rand', 
                        axis='all', 
                        v_scale=(0.5, 2), 
                        col=_PETRainbowCMAP, 
                        doses=None, 
                        metrics=None):
    """[summary]

    Args:
        data ([type]): [description]
        patient_id ([type]): [description]
        file_out ([type]): [description]
        slices (str, optional): [description]. Defaults to 'rand'.
        axis (str, optional): [description]. Defaults to 'all'.
        v_scale (tuple, optional): [description]. Defaults to (0.5, 2).
        col ([type], optional): [description]. Defaults to PETRainbowCMAP.
        doses ([type], optional): [description]. Defaults to None.
        metrics ([type], optional): [description]. Defaults to None.
    """    
    if slices == 'rand':
        slices = [np.random.randint(55, 75) for _ in range(3)]

    if doses:
        if not isinstance(doses[0], str):
            doses = [str(int(d)) + ' MBq' for d in doses]

    num_rows = len(slices)
    num_cols = 3 #data.shape[0]
    
    # plot_mask_contour = False
    # if data.shape[0] > 3:
    #     plot_mask_contour = True
        
    fig, ax = plt.subplots(num_rows,
                           num_cols,
                           figsize=(num_cols*4, num_rows*4),
                           gridspec_kw={'wspace': 0, 'hspace': 0})
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    plt.margins(0.1)
    
    # normalization
    # normalization = _MidPointNorm(vmin=-1, 
    #                               vmax=0.9*data[1].max(),
    #                               midpoints = [0],
    #                               defmids=[0.07])

    titles = ['LOW DOSE', 'HIGH DOSE', 'INFERRED']

    for idx, slice_i in enumerate(slices):
        if axis == 'all':
            axis = idx
        for idy in range(num_cols):
            # extract 2d image from 3D volume given a slice number
            data2d = data[idy].take(indices=slice_i, axis=axis)
            data2d = np.rot90(data2d)
            im = ax[idx, idy].imshow(
                data2d, cmap=col, aspect='equal', vmin=v_scale[0], vmax=v_scale[1]) #, norm=normalization)
            
            if segment_data.any():
                for j in range(0, segment_data.shape[0]):
                    mask2d = segment_data[j].take(indices=slice_i, axis=axis)
                    mask2d = (mask2d > 0).astype(int)
                    if mask2d.any():
                        mask2d = np.rot90(mask2d)
                        colors = 'yellow' if j == segment_data.shape[0] - 1 else 'white'
                        styles = {'colors': colors, 'linewidths': 0.8} #, 'linestyles': 'dotted'}
                        ax[idx, idy].contour(mask2d, **styles)
                        
            ax[idx, idy].axis('off')         
            
            # place text on figure
            plot_extent = np.max(data2d.shape)
            if idx == num_rows - 1:
                ax[idx, idy].text(0.03*plot_extent, 0.96*plot_extent, titles[idy],
                                  color='white', fontsize=14)
                if idy in [0, 1] and doses:
                    ax[idx, idy].text(0.03*plot_extent, 0.88*plot_extent, doses[idy],
                                      color='white', fontsize=14)
            if idy == 0:
                ax[idx, idy].text(0.03*plot_extent, 0.16*plot_extent, str(slice_i),
                                  color='white', fontsize=14)
    
    # patient id
    ax[0, 0].text(0.03*plot_extent, 0.08*plot_extent, patient_id,
                    color='white', fontsize=14)
    # CP ratio
    if metrics:
        ax[0, 0].text(0.05*plot_extent, 0.96*plot_extent, f"Cortical uptake ratio (SUVr) = {np.round(metrics['lowdose'], 2)}",
                        color='white', fontsize=14)
        ax[0, 1].text(0.75*plot_extent, 0.96*plot_extent, np.round(metrics['true'], 2),
                        color='white', fontsize=14)
        ax[0, 2].text(0.75*plot_extent, 0.96*plot_extent, np.round(metrics['infer'], 2),
                        color='white', fontsize=14)

    cbaxes = inset_axes(ax[0, 2], width="90%", height="5%", loc=1)
    cbar = fig.colorbar(im, cax=cbaxes, orientation='horizontal')
    cbar.ax.tick_params(colors='white', labelsize=14)
    # cbar.set_ticks([])
    plt.savefig(file_out, bbox_inches='tight')
    plt.close('all')

def brain_slices_grid_pe2i(data, 
                      patient_id, 
                      file_out,
                      segment_data=None,
                      slices='rand', 
                      axis='all', 
                      v_scale=(0.5, 2), 
                      col=_PETRainbowCMAP, 
                      doses=None, 
                      metrics=None):
    """[summary]

    Args:
        data ([type]): [description]
        patient_id ([type]): [description]
        file_out ([type]): [description]
        slices (str, optional): [description]. Defaults to 'rand'.
        axis (str, optional): [description]. Defaults to 'all'.
        v_scale (tuple, optional): [description]. Defaults to (0.5, 2).
        col ([type], optional): [description]. Defaults to PETRainbowCMAP.
        doses ([type], optional): [description]. Defaults to None.
        metrics ([type], optional): [description]. Defaults to None.
    """    
    if slices == 'rand':
        slices = [np.random.randint(55, 75) for _ in range(3)]

    if doses:
        if not isinstance(doses[0], str):
            doses = [str(int(d)) + ' MBq' for d in doses]

    num_rows = len(slices)
    num_cols = 3 #data.shape[0]
        
    fig, ax = plt.subplots(num_rows,
                           num_cols,
                           figsize=(num_cols*4, num_rows*4),
                           gridspec_kw={'wspace': 0, 'hspace': 0})
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    plt.margins(0.1)
    
    # normalization
    normalization = _MidPointNorm(vmin=-1, 
                                  vmax=0.9*data[1].max(),
                                  midpoints = [0],
                                  defmids=[0.07])

    titles = ['LOW DOSE', 'HIGH DOSE', 'INFERRED']

    for idx, slice_i in enumerate(slices):
        if axis == 'all':
            axis = idx
        for idy in range(num_cols):
            # extract 2d image from 3D volume given a slice number
            data2d = data[idy].take(indices=slice_i, axis=axis)
            data2d = np.rot90(data2d)
            im = ax[idx, idy].imshow(
                data2d, cmap=col, aspect='equal', norm=normalization) #vmin=v_scale[0], vmax=v_scale[1], norm=)
                    
            if segment_data.any():
                for j in range(0, segment_data.shape[0]):
                    mask2d = segment_data[j].take(indices=slice_i, axis=axis)
                    mask2d = (mask2d > 0).astype(int)
                    if mask2d.any():
                        mask2d = np.rot90(mask2d)
                        colors = 'yellow' if j == segment_data.shape[0] - 1 else 'white'
                        styles = {'colors': colors, 'linewidths': 0.8} #, 'linestyles': 'dotted'}
                        ax[idx, idy].contour(mask2d, **styles)
                
            ax[idx, idy].axis('off')         
            
            # place text on figure
            plot_extent = np.max(data2d.shape)
            if idx == num_rows - 1:
                ax[idx, idy].text(0.03*plot_extent, 0.96*plot_extent, titles[idy],
                                  color='white', fontsize=14)
                if idy in [0, 1] and doses:
                    ax[idx, idy].text(0.03*plot_extent, 0.88*plot_extent, doses[idy],
                                      color='white', fontsize=14)
            if idy == 0:
                ax[idx, idy].text(0.03*plot_extent, 0.16*plot_extent, str(slice_i),
                                  color='white', fontsize=14)
    
    # patient id
    ax[0, 0].text(0.03*plot_extent, 0.08*plot_extent, patient_id,
                    color='white', fontsize=14)
    # CP ratio
    ax[0, 0].text(0.6*plot_extent, 0.96*plot_extent, f"r_CP = {np.round(metrics['lowdose'], 2)}",
                    color='white', fontsize=14)
    ax[0, 1].text(0.75*plot_extent, 0.96*plot_extent, np.round(metrics['true'], 2),
                    color='white', fontsize=14)
    ax[0, 2].text(0.75*plot_extent, 0.96*plot_extent, np.round(metrics['infer'], 2),
                    color='white', fontsize=14)

    cbaxes = inset_axes(ax[0, 2], width="90%", height="5%", loc=1)
    cbar = fig.colorbar(im, cax=cbaxes, orientation='horizontal')
    cbar.ax.tick_params(colors='white', labelsize=14)
    # cbar.set_ticks([])
    plt.savefig(file_out, bbox_inches='tight')
    plt.close('all')
    
    
# def pixelvalue_jointplot(x, y, fname, ylabel='Pixel value', color='tab:blue'):
    
#     """Generate and saves joint plot of pixel values between two images

#     Args:
#         x (np.ndarray): Reference image matrix (to compare to)
#         y (np.ndarray): Second image matrix
#         fname (str or path-like object): full path name where to save image
#         ylabel (str, optional): Y-axis label. Defaults to 'Pixel value'.
#     """
    
#     true_mask = np.where(x != 0)
#     x = x[true_mask]
#     y = y[true_mask]

#     plt.figure(figsize=(6, 6))
#     ax = sns.histplot(x=x, y=y, cmap='plasma', binwidth=100)
#     ax.set_facecolor('lightgray')#, space=0, height=8, joint_kws=joint_kws, marginal_kws=marginal_kws)
#     xl = ax.get_xlim()[1]
#     ax.plot([0, xl], [0, xl], color='lightgreen', linewidth=1.5)
#     ax.set_xlim(0, xl)
#     ax.set_ylim(0, xl)
#     ax.set_xlabel('Pixel value (ref image)', fontsize=16)
#     ax.set_ylabel(ylabel, fontsize=16)
#     plt.savefig(fname, bbox_inches='tight')
#     plt.close('all')
    
    
# def pixelvalue_jointplot_old(x, y, fname, ylabel='Pixel value', color='tab:blue'):
    
#     """Generate and saves joint plot of pixel values between two images

#     Args:
#         x (np.ndarray): Reference image matrix (to compare to)
#         y (np.ndarray): Second image matrix
#         fname (str or path-like object): full path name where to save image
#         ylabel (str, optional): Y-axis label. Defaults to 'Pixel value'.
#     """
    
#     true_mask = np.where(x != 0)
#     x = x[true_mask]
#     y = y[true_mask]

#     plt.figure(figsize=(15, 15))
#     joint_kws={'alpha': 0.3, 'marker': '.'}
#     marginal_kws={'kde': True}
#     # x = x.flatten()
#     # x[x == 0 ] = np.nan
#     # y = y.flatten()
#     # y[y == 0 ] = np.nan
#     jp = sns.jointplot(x=x.flatten(), y=y.flatten(), space=0, height=8, joint_kws=joint_kws, marginal_kws=marginal_kws, color=color)
#     jp.ax_joint.plot([0,1], [0,1], '-r', transform=jp.ax_joint.transAxes)
#     jp.set_axis_labels('Pixel value (ref image)', ylabel, fontsize=16)
#     plt.savefig(fname, bbox_inches='tight')
#     plt.close('all')

# def pixelvalue_jointplot(x_ld, x_true, x_infer, fname='', bins=50, crop_th=0.5): #fname, ylabel='Pixel value', color='tab:blue'):
       
#     true_mask = np.where(x_true > 0.1)
#     x_true = x_true[true_mask]
#     x_ld = x_ld[true_mask]
#     x_infer = x_infer[true_mask]
#     crop = np.where(x_true < crop_th * np.max(x_true))
    
#     fig = plt.figure(figsize=(6, 6))

#     gs = GridSpec(2, 2, figure=fig)
#     ax1 = fig.add_subplot(gs[0, 0])
#     ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
#     ax3 = fig.add_subplot(gs[1,:])
    
#     # AX1
#     sns.histplot(ax=ax1, x=x_true[crop], y=x_ld[crop], cmap='plasma', bins=bins)
#     ax1.set_facecolor('k')#, space=0, height=8, joint_kws=joint_kws, marginal_kws=marginal_kws)
#     xl = ax1.get_xlim()[1]
#     ax1.plot([0, xl], [0, xl], color='lightgreen', linewidth=1.5)
#     ax1.set_xlim(0, .95*xl)
#     ax1.set_ylim(0, .95*xl)
#     ax1.text(0.05*xl, 0.85*xl, "(a) LOW DOSE", color='w')
#     ax1.xaxis.set_ticklabels([])
#     ax1.yaxis.set_ticklabels([])
#     ax1.set_ylabel('Pixel value')
#     ax1.set_xlabel('                                     Pixel value (ref image)')
# #     ax1.set_ylabel('Pixel value (low dose)')
    
#     # AX2
#     sns.histplot(ax=ax2, x=x_true[crop], y=x_infer[crop], cmap='plasma', bins=bins)
#     ax2.set_facecolor('k')#, space=0, height=8, joint_kws=joint_kws, marginal_kws=marginal_kws)
#     xl = ax2.get_xlim()[1]
#     ax2.plot([0, xl], [0, xl], color='lightgreen', linewidth=1.5)
#     ax2.set_xlim(0, .95*xl)
#     ax2.set_ylim(0, .95*xl)
#     ax2.text(0.05*xl, 0.85*xl, "(b) DE-NOISED", color='w')
#     ax2.xaxis.set_ticklabels([])
#     ax2.yaxis.set_ticklabels([])
# #     ax2.set_xlabel('Pixel value (ref image)')
# #     ax2.set_ylabel('Pixel value (de-noised)')
    
#     # DIFF PLOT - AX3
#     pd_ld_true = (x_ld - x_true) / x_true * 100
#     pd_infer_true = (x_infer - x_true) / x_true * 100
#     # sns.kdeplot(x=x_ld.flatten(), label='lowdose', fill=True, color="blue", alpha=.1)
#     sns.histplot(ax=ax3, x=pd_ld_true, label='low dose', color=colors[0], fill=True, alpha=.2, binwidth=2, edgecolor=None)
#     sns.histplot(ax=ax3, x=pd_infer_true, label='de-noised', color=colors[1], fill=True, alpha=.2, binwidth=2, edgecolor=None)
#     ax3.set_xlim(-100, 100)
#     ax3.legend()
#     ax3.set_xlabel("Pixel value %-diff")
#     ax3.set_ylabel("Count")
#     xl2 = ax3.get_ylim()[1]
#     ax3.text(-95, 0.9*xl2, "(c)")
#     ax3.yaxis.set_ticklabels([])
#     plt.savefig(fname, format='pdf', bbox_inches='tight')
#     plt.close('all')


def pixelvalue_jointplot_log(x_ld, x_true, x_infer, hist1, hist2, fname='', nbins=200, crop_th=0.5): #fname, ylabel='Pixel value', color='tab:blue'):
       
    true_mask = np.where(x_true > 0.1)
    x_true = x_true[true_mask]
    x_ld = x_ld[true_mask]
    x_infer = x_infer[true_mask]
    # crop = np.where(x_true < crop_th * np.max(x_true))
    
    fig = plt.figure(figsize=(6, 6))

    gs = GridSpec(2, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    ax3 = fig.add_subplot(gs[1,:])
    
    # AX1
    hist1c = hist1.copy()
    hist1c[hist1>0] = np.log10(hist1[hist1>0])
    hist1c[hist1==0] = np.min(hist1c)

    ax1.imshow(hist1c.transpose(), cmap=_PETRainbowCMAP, origin='lower')
    ax1.plot([nbins,0],[nbins,0],'-',color="lightgreen", linewidth=1.5)
    # sns.histplot(ax=ax1, x=x_true[crop], y=x_ld[crop], cmap='plasma', bins=bins)
    # ax1.set_facecolor('k')#, space=0, height=8, joint_kws=joint_kws, marginal_kws=marginal_kws)
    xl = ax1.get_xlim()[1]
    # ax1.plot([0, xl], [0, xl], color='lightgreen', linewidth=1.5)
    ax1.set_xlim(0, .95*xl)
    ax1.set_ylim(0, .95*xl)
    ax1.text(0.05*xl, 0.85*xl, "(a) LOW ACTIVITY", color='w')
    ax1.xaxis.set_ticklabels([])
    ax1.yaxis.set_ticklabels([])
    ax1.set_ylabel('Pixel value')
    ax1.set_xlabel('                                     Pixel value (ref image)')
#     ax1.set_ylabel('Pixel value (low dose)')
    
    # AX2
    hist2c = hist2.copy()
    hist2c[hist2>0] = np.log10(hist2[hist2>0])
    hist2c[hist2==0] = np.min(hist2c)

    im = ax2.imshow(hist2c.transpose(), cmap=_PETRainbowCMAP, origin='lower')
    ax2.plot([nbins,0],[nbins,0],'-',color="lightgreen", linewidth=1.5)
    # sns.histplot(ax=ax2, x=x_true[crop], y=x_infer[crop], cmap='plasma', bins=bins)
    # ax2.set_facecolor('k')#, space=0, height=8, joint_kws=joint_kws, marginal_kws=marginal_kws)
    xl = ax2.get_xlim()[1]
    # ax2.plot([0, xl], [0, xl], color='lightgreen', linewidth=1.5)
    ax2.set_xlim(0, .95*xl)
    ax2.set_ylim(0, .95*xl)
    ax2.text(0.05*xl, 0.85*xl, "(b) DE-NOISED", color='w')
    ax2.xaxis.set_ticklabels([])
    ax2.yaxis.set_ticklabels([])
    cbaxes = inset_axes(ax2, width="40%", height="3%", loc=4)
    cbar = fig.colorbar(im, cax=cbaxes, orientation='horizontal')
    cbar.ax.tick_params(colors='white', labelsize=10)
#     ax2.set_xlabel('Pixel value (ref image)')
#     ax2.set_ylabel('Pixel value (de-noised)')
    
    # DIFF PLOT - AX3
    pd_ld_true = (x_ld - x_true) / x_true * 100
    pd_infer_true = (x_infer - x_true) / x_true * 100
    # sns.kdeplot(x=x_ld.flatten(), label='lowdose', fill=True, color="blue", alpha=.1)
    sns.histplot(ax=ax3, x=pd_ld_true, label='low activity', color=colors[0], fill=True, alpha=.2, binwidth=2, edgecolor=None)
    sns.histplot(ax=ax3, x=pd_infer_true, label='de-noised', color=colors[1], fill=True, alpha=.2, binwidth=2, edgecolor=None)
    ax3.set_xlim(-100, 100)
    ax3.legend()
    ax3.set_xlabel("Pixel value %-diff")
    ax3.set_ylabel("Count")
    xl2 = ax3.get_ylim()[1]
    ax3.text(-95, 0.9*xl2, "(c)")
    ax3.yaxis.set_ticklabels([])
    plt.savefig(fname, format='pdf', bbox_inches='tight')
    plt.close('all')
    
    
# def pixelvalue_hist_kde(x_ld, x_true, x_infer, mask, fname):
    
#     """Computes % diff of pixel value between lowdose-true image and  inferred-true image.

#     Args:
#         x_ld ([type]): [description]
#         x_true ([type]): [description]
#         x_infer ([type]): [description]
#         fname ([type]): [description]
#     """  
    
#     # true_mask = np.where(x_true != 0)
#     x_ld = x_ld[mask]
#     x_true = x_true[mask]
#     x_infer = x_infer[mask]
#     pd_ld_true = (x_ld - x_true) / x_true * 100
#     pd_infer_true = (x_infer - x_true) / x_true * 100
#     plt.figure(figsize=(18, 7))
#     # sns.kdeplot(x=x_ld.flatten(), label='lowdose', fill=True, color="blue", alpha=.1)
#     sns.histplot(x=pd_ld_true, label='lowdose - true', fill=True, color=colors[0], alpha=.2, binwidth=2, kde=True)
#     ax = sns.histplot(x=pd_infer_true, label='AI - true', fill=True, color=colors[1], alpha=.2, binwidth=2, kde=True)
#     plt.xlim([-100, 100])
#     ax.set_xlabel("Pixel value %-diff", fontsize=16)
#     ax.set_ylabel("Kernel density est.", fontsize=16)
#     ax.tick_params(labelsize=14)
#     plt.legend(fontsize=16)
#     plt.savefig(fname, bbox_inches='tight')
#     plt.close('all')
            
            
def box_roi_percent_diff_images(x_ld, x_true, x_infer, fname, extent=0):
    
    """[summary]

    Args:
        x_ld ([type]): [description]
        x_true ([type]): [description]
        x_infer ([type]): [description]
        fname ([type]): [description]
    """    
    
    fig000 = plt.figure(figsize=(12, 4))
    ax11, ax21, ax31 = ImageGrid(fig000, 111,          # as in plt.subplot(111)
                    nrows_ncols=(1,3),
                    axes_pad=0.15,
                    share_all=True,
                    cbar_location="right",
                    cbar_mode="single",
                    cbar_size="7%",
                    cbar_pad=0.15,
                    )

    soi = int(x_true.shape[1] / 2)
    x_true[x_true == 0] = 1
    percent_diff_lowdose_true_full = (x_ld - x_true) / x_true *100
    percent_diff_infer_true_full = (x_infer - x_true) / x_true * 100

    y_text_pos = 128 - extent + 5
    ax11.imshow(x_true[:,soi,:], cmap=_PETRainbowCMAP, aspect='equal', vmin=0, vmax=x_true[:,soi,:].max()*0.99)
    ax11.set_xlim([soi-extent, soi+extent])
    ax11.set_ylim([soi-extent, soi+extent])
    ax11.text(135, y_text_pos, 'fulldose ref', color='white', fontsize=14)

    ax21.imshow(percent_diff_lowdose_true_full[:,soi,:], cmap='hot', aspect='equal', vmin=0, vmax=100)
    ax21.set_xlim([soi-extent, soi+extent])
    ax21.set_ylim([soi-extent, soi+extent])
    ax21.text(125, y_text_pos, '%-diff lowdose', color='white', fontsize=14)


    im=ax31.imshow(percent_diff_infer_true_full[:,soi,:], cmap='hot', aspect='equal', vmin=0, vmax=100)
    ax31.set_xlim([soi-extent, soi+extent])
    ax31.set_ylim([soi-extent, soi+extent])
    ax31.text(125, y_text_pos, '%-diff inferred', color='white', fontsize=14)

    # Colorbar
    ax31.cax.colorbar(im)
    ax31.cax.toggle_label(True)
    plt.savefig(fname, bbox_inches='tight')
    plt.close('all')
    
    
def bland_altman_plot(df, dirname, fname='', xlim=None, ylim=None):
    data = df.copy()
    data['mean_true_ld'] = (data['lowdose'] + data['true']) / 2
    data['diff_true_ld'] = data['lowdose'] - data['true']
    data['mean_true_infer'] = (data['infer'] + data['true']) / 2
    data['diff_true_infer'] = data['infer'] - data['true']
    md = np.mean(data['diff_true_ld'].values)
    sd = np.std(data['diff_true_ld'].values)

    plt.figure(figsize=(9,6))
    ax = sns.scatterplot(x=data['mean_true_ld'], y=data['diff_true_ld'], label='lowdose - true', color=colors[0])
    sns.scatterplot(x=data['mean_true_infer'], y=data['diff_true_infer'], label='AI - true', color=colors[1])
    plt.axhline(md,           color='gray', linestyle='--')
    plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
    ax.set_xlabel(fname, fontsize=16)
    ax.set_ylabel("Difference", fontsize=16)
    ax.tick_params(labelsize=14)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    plt.legend(fontsize=16)
    figname = dirname.joinpath(fname + "_bland_altman.png")
    plt.savefig(figname, bbox_inches='tight')
    plt.close('all')
    
    
def boxplot_from_dataframe(df, dirname, fname, orient='h'):
    # csvname = dirname.joinpath(fname + ".csv")
    # df.to_csv(csvname)
    
    plt.figure(figsize=(10, 7))
    ax = sns.barplot(data=df, y='metric', x='value', orient=orient, hue='phase', palette=palette)
    ax.set_xlabel('Percent difference', fontsize=16)
    ax.set_ylabel('Variables', fontsize=16)
    ax.tick_params(labelsize=14)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[0:], labels=labels[0:], frameon=False, fontsize=16)
    figname = dirname.joinpath(fname + ".png")
    plt.savefig(figname, bbox_inches='tight')
    plt.close('all')
    
    
def ssmi_psnr_nrmse_plot(df, dirname, fname):
    csvname3b = dirname.joinpath(fname + ".csv")
    df.to_csv(csvname3b)
    
    _, ax = plt.subplots(1, 3, figsize=(16, 6), squeeze=False)
    for j, kw in enumerate(['ssmi', 'nrmse', 'psnr']):
        sns.boxplot(ax=ax[0, j], data=df[df['metric']==kw], x='metric', y='value', hue='phase', palette=palette)
        # ax[0, j].set_title(kw.upper(), fontsize=16)
        handles, labels = ax[0, j].get_legend_handles_labels()
        if j == 1:
            ax[0, j].legend(handles=handles[0:], labels=labels[0:], frameon=False, fontsize=14)
        else:
            ax[0, j].legend(labels=[], frameon=False)
        ax[0, j].set_xlabel('', fontsize=16)
        ax[0, j].set_ylabel('', fontsize=16)
        ax[0, j].tick_params(labelsize=14)
    
    figname = dirname.joinpath(fname + ".png")
    plt.savefig(figname, bbox_inches='tight')
    plt.close('all')
        
        
# def hist_kde_percent_diff(df, dirname, fname):
#     data = df.copy() 
#     data['percent_diff'] = data.apply(lambda row: 100*np.abs(row['infer']-row['true']) / row['true'], axis=1)
#     csvname = dirname.joinpath(fname + ".csv")
#     data.to_csv(csvname)
#     # hist/kde plot of percent diff PC ratio
#     plt.figure(figsize=(10, 7))
#     ax5 = sns.histplot(data=data, x='percent_diff', kde=True)
#     ax5.set_xlabel("% diff " + fname, fontsize=16)
#     ax5.tick_params(labelsize=14)
#     figname5 = dirname.joinpath(fname + "_percent_diff.png")
#     plt.savefig(figname5, bbox_inches='tight')
#     plt.close('all')
    
    
# def plot_unstack_metrics_barplot(df, dirname, fname):
#     dfu = df.unstack().reset_index()
#     dfu.columns = ['mode', 'patient', fname]

#     plt.figure(figsize=(12, 16))
#     ax4 = sns.barplot(y='patient', x=fname, data=dfu, hue='mode', orient='h', palette='rainbow', errwidth=1.5)
#     ax4.set_xlabel(fname, fontsize=16)
#     ax4.set_ylabel("Patient", fontsize=16)
#     ax4.legend(fontsize=16)
#     ax4.tick_params(labelsize=14)
#     figname = dirname.joinpath(fname + "_all_patients.png")
#     plt.savefig(figname, bbox_inches='tight')
#     plt.close('all')
    
    
def lmplot_compare(df, dirname, fname, thresh=None, xlim=None, ylim=None):
        
    plt.figure(figsize=(7, 7))
    lr_ai = lr_model(df['true'], df['infer'])
    lr_ld = lr_model(df['true'], df['lowdose'])
    sns.regplot(data=df, x="true", y='lowdose', color=colors[0], label=f'low-dose: {lr_ld[0]} +/- {lr_ld[1]}')
    ax = sns.regplot(data=df, x="true", y='infer', color=colors[1], label=f'AI: {lr_ai[0]} +/- {lr_ai[1]}')
    if thresh:
        ax.plot([thresh, thresh], ax.get_ylim(), 'k--', linewidth=1)
        ax.plot(ax.get_xlim(), [thresh, thresh], 'k--', linewidth=1)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    ax.set_xlabel(f"Ref. {fname[:-3]} (SUVr)", fontsize=16)
    ax.set_ylabel(f"{fname[:-3]} (SUVr)", fontsize=16)
    # plt.label(['AI-inferred', 'low-dose'])
    ax.tick_params(labelsize=14)
    plt.legend(fontsize=16, frameon=False)
    figname = dirname.joinpath(fname + "_all.png")
    plt.savefig(figname, bbox_inches='tight')
    plt.close('all')