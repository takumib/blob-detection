/*
 * Copyright (C) 2008, 2009, 2010 Richard Membarth <richard.membarth@cs.fau.de>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with NVIDIA CUDA Software Development Kit (or a modified version of that
 * library), containing parts covered by the terms of a royalty-free,
 * non-exclusive license, the licensors of this Program grant you additional
 * permission to convey the resulting work.
 */


// This looks to be highly modifiable 

#include <inttypes.h>
#include <sys/time.h>

#include <libgimp/gimp.h>
#include <libgimp/gimpui.h>

#include "gimp_main.hpp"
#include "gimp_gui.hpp"
#include "blob_definitions.h"
#include "blob_host.hpp"

static void query(void);
static void run(const gchar *name, gint nparams, 
				const GimpParam  *param, gint *nreturn_vals,
                GimpParam  **return_vals);

void run_multi_filter(GimpDrawable *drawable, GimpPreview  *preview);

filter_config filter_vals =
{
};

GimpPlugInInfo PLUG_IN_INFO =
{
    NULL,
    NULL,
    query,
    run
};

MAIN()

static void query (void) 
{
 	static GimpParamDef args[] =
    {
		{GIMP_PDB_INT32, (gchar *)"run-mode", (gchar *)"Run mode"},
		{GIMP_PDB_IMAGE, (gchar *)"image", (gchar *)"Input image"},
		{GIMP_PDB_DRAWABLE,	"drawable",	(gchar *)"Input drawable"},
		{GIMP_PDB_INT32, (gchar *)"gpu", (gchar *)"Run on GPU"},
		{GIMP_PDB_INT32, (gchar *)"gpu_op", (gchar *)"Run on GPU With Textures"},
		//{GIMP_PDB_INT32, (gchar *)"gpu_shared_op", (gchar *)"Run on GPU Shared Optimized"}
	};
    
    gimp_install_procedure(
        "plug-in-blob",
        "Blob Detection...",
        "Detects the presence of blobs",
        "Takumi Bolte (sttjb05@moravian.edu)",
        "Takumi Bolte",
        "06/22/2012",
        "_Blob...",
        "GRAY*",
        GIMP_PLUGIN,
        G_N_ELEMENTS(args), 0,
        args, NULL);
    
    gimp_plugin_menu_register("plug-in-blob", "<Image>/Filters/GPU Filters/Blob");
}
    

static void run(const gchar *name, gint nparams, const GimpParam *param, gint *nreturn_vals, GimpParam **return_vals) 
{
    static GimpParam  values[1];
    GimpPDBStatusType status = GIMP_PDB_SUCCESS;
    GimpRunMode       run_mode;
    GimpDrawable     *drawable;
    
    /* Setting mandatory output values */
    *nreturn_vals = 1;
    *return_vals  = values;
    
    values[0].type = GIMP_PDB_STATUS;
    values[0].data.d_status = status;
    
    /* Getting run_mode - we won't display a dialog if 
     * we are in NONINTERACTIVE mode */
    run_mode = (GimpRunMode)param[0].data.d_int32;
    
    /*  Get the specified drawable  */
    drawable = gimp_drawable_get(param[2].data.d_drawable);
    
    switch (run_mode) 
	{
        case GIMP_RUN_INTERACTIVE:
            /* Get options last values if needed */
            gimp_get_data("plug-in-blob", &filter_vals);
            
            /* Display the dialog */
            if (!multi_res_dialog(drawable))
                return;
            break;
        
        case GIMP_RUN_NONINTERACTIVE:

        case GIMP_RUN_WITH_LAST_VALS:
            /*  Get options last values if needed  */
            gimp_get_data("plug-in-blob", &filter_vals);
            break;
        
        default:
            break;
    }
   
	wakeup();
 
    run_multi_filter(drawable, NULL);
    
    gimp_displays_flush();
    gimp_drawable_detach(drawable);
    
    /*  Finally, set options in the core  */
    if (run_mode == GIMP_RUN_INTERACTIVE)
      gimp_set_data("plug-in-blob", &filter_vals, sizeof(filter_config));
    
    return;
}
    

void run_multi_filter(GimpDrawable *drawable, GimpPreview  *preview) 
{
    gint         x1, y1, x2, y2;
    GimpPixelRgn rgn_in, rgn_out;
    guchar       *g0;
    gint         width, height, channels;
   
	timeval start, end;
	double elapsedtime;
 
    if (!preview)
        gimp_progress_init("Multiresolution filter...");
    
    /* Gets upper left and lower right coordinates,
     * and layers number in the image */
    if (preview) 
	{
        gimp_preview_get_position(preview, &x1, &y1);
        gimp_preview_get_size(preview, &width, &height);
        x2 = x1 + width;
        y2 = y1 + height;
    } 
	else 
	{
        gimp_drawable_mask_bounds(drawable->drawable_id, &x1, &y1, &x2, &y2);
        width = x2 - x1;
        height = y2 - y1;
    }
    
    channels = gimp_drawable_bpp(drawable->drawable_id);
    
    /* Initialises two PixelRgns, one to read original data,
     * and the other to write output data. That second one will
     * be merged at the end by the call to
     * gimp_drawable_merge_shadow() */
    gimp_pixel_rgn_init(&rgn_in,
                         drawable,
                         x1, y1,
                         width, height,
                         FALSE, FALSE);
    gimp_pixel_rgn_init(&rgn_out,
                         drawable,
                         x1, y1,
                         width, height,
                         preview == NULL, TRUE);
    
    /* Allocate memory for input and output image */
    g0 = g_new(guchar, width*height*channels);
    
    gimp_pixel_rgn_get_rect(&rgn_in, g0, x1, y1, width, height);

	if(filter_vals.gpu)
	{
		gettimeofday(&start, NULL);
			
		run_gpu(g0, width, height);

		gettimeofday(&end, NULL);

		elapsedtime = (end.tv_sec - start.tv_sec) * 1000.0;
		elapsedtime += (end.tv_usec - start.tv_usec) / 1000.0;

		printf("GPU Time: %.2f ms.\n", elapsedtime);
	}
	else if(filter_vals.gpu_op)
	{
		gettimeofday(&start, NULL);
			
		tex_run_gpu(g0, width, height);

		gettimeofday(&end, NULL);

		elapsedtime = (end.tv_sec - start.tv_sec) * 1000.0;
		elapsedtime += (end.tv_usec - start.tv_usec) / 1000.0;

		printf("GPU With Textures Time: %.2f ms.\n", elapsedtime);
	}
	/*else if(filter_vals.gpu_shared_op)
	{
		gettimeofday(&start, NULL);

		run_gpu_shared_op(g0, width, height);

		gettimeofday(&end, NULL);

		elapsedtime = (end.tv_sec - start.tv_sec) * 1000.0;
		elapsedtime += (end.tv_usec - start.tv_usec) / 1000.0;

		printf("GPU Shared Optimized Time: %.2f ms.\n", elapsedtime);
	}*/
	else {
		gettimeofday(&start, NULL);

		run_cpu(g0, width, height, channels);

		gettimeofday(&end, NULL);

		elapsedtime = (end.tv_sec - start.tv_sec) * 1000.0;
		elapsedtime += (end.tv_usec - start.tv_usec) / 1000.0;

		printf("CPU Time: %.2f ms.\n", elapsedtime);
	}

	//multi_res_cpu.cpp
    //run_cpu(g0, width, height, channels);
    
	//run_gpu(g0, width, height);
 
    gimp_pixel_rgn_set_rect(&rgn_out, g0, x1, y1, width, height);
    g_free(g0);
    
    /*  Update the modified region */
    if (preview) 
	{
        gimp_drawable_preview_draw_region(GIMP_DRAWABLE_PREVIEW(preview), &rgn_out);
    } 
	else 
	{
        gimp_drawable_flush(drawable);
        gimp_drawable_merge_shadow(drawable->drawable_id, TRUE);
        gimp_drawable_update(drawable->drawable_id, x1, y1, width, height);
    }
}

