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

#include <libgimp/gimp.h>
#include <libgimp/gimpui.h>

#include "gimp_main.hpp"
#include "gimp_gui.hpp"
#include "blob_definitions.h"
#include "blob_host.hpp"

gboolean multi_res_dialog (GimpDrawable *drawable) {
    GtkWidget *dialog;
    GtkWidget *main_vbox;
    GtkWidget *preview;
    gboolean   run;

	GtkWidget *gpu_button;
   	GtkWidget *gpu_tex_button;
	GtkWidget *gpu_shared_op_button;
 
    gimp_ui_init("Generic Blob Detector", FALSE);
    
    dialog = gimp_dialog_new("Generic Blob Detector", "Generic Blob Detector",
                              NULL, (GtkDialogFlags) 0,
                              gimp_standard_help_func, "plug-in-blob",
                              GTK_STOCK_CANCEL, GTK_RESPONSE_CANCEL,
                              GTK_STOCK_OK,     GTK_RESPONSE_OK,
                              NULL);
    
    main_vbox = gtk_vbox_new(FALSE, 6);
    gtk_container_add(GTK_CONTAINER(GTK_DIALOG (dialog)->vbox), main_vbox);
    gtk_widget_show(main_vbox);
   
	gpu_button = gtk_check_button_new_with_mnemonic("_Use GPU");
	gtk_box_pack_start(GTK_BOX(main_vbox), gpu_button, FALSE, FALSE, 0);

	gpu_tex_button = gtk_check_button_new_with_mnemonic("_Use GPU With Textures");
	gtk_box_pack_start(GTK_BOX(main_vbox), gpu_tex_button, FALSE, FALSE, 0);

	//gpu_shared_op_button = gtk_check_button_new_with_mnemonic("_Use GPU Shared Optimized");
	//gtk_box_pack_start(GTK_BOX(main_vbox), gpu_shared_op_button, FALSE, FALSE, 0);

	if(has_cuda_device())
	{
		gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(gpu_button), TRUE);
		gtk_widget_set_sensitive(gpu_button, TRUE);
		filter_vals.gpu = TRUE;

	    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(gpu_tex_button), TRUE);
		gtk_widget_set_sensitive(gpu_tex_button, TRUE);
	    filter_vals.gpu_op = TRUE;	

		//gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(gpu_shared_op_button), TRUE);
		//gtk_widget_set_sensitive(gpu_shared_op_button, TRUE);
		//filter_vals.gpu_shared_op = TRUE;
	}
	else
	{
		gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(gpu_button), FALSE);
		gtk_widget_set_sensitive(gpu_button, FALSE);
		filter_vals.gpu = FALSE;

		gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(gpu_tex_button), FALSE);
		gtk_widget_set_sensitive(gpu_tex_button, FALSE);
		filter_vals.gpu_op = FALSE;

		//gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(gpu_shared_op_button), FALSE);
		//filter_vals.gpu_shared_op = FALSE;
	}
 
	g_signal_connect(gpu_button, "toggled", G_CALLBACK(gimp_toggle_button_update), &filter_vals.gpu);

	g_signal_connect(gpu_tex_button, "toggled", G_CALLBACK(gimp_toggle_button_update), &filter_vals.gpu_op);

	//g_signal_connect(gpu_shared_op_button, "toggled", G_CALLBACK(gimp_toggle_button_update), &filter_vals.gpu_shared_op);

	gtk_widget_show(gpu_button);
	gtk_widget_show(gpu_tex_button);
	//gtk_widget_show(gpu_shared_op_button);
	
    gtk_widget_show(dialog);
    
    //run_multi_filter(drawable, GIMP_PREVIEW(preview));
    
    run = (gimp_dialog_run(GIMP_DIALOG(dialog)) == GTK_RESPONSE_OK);
    
    gtk_widget_destroy(dialog);
    
    return run;
}

