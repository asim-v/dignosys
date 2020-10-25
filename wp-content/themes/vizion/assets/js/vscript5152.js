jQuery(document).ready( function() {
    jQuery(".job-search").click( function(e) {
        e.preventDefault(); 
        var search_job   = jQuery("#search_job").val();
        var category     = jQuery("#job_category").val();
        
        jQuery.ajax({

            type : "post",          
            url : job_finder.ajaxurl,
            data : {
                    action: "job_filter_search", 
                    search_job : search_job,
                    category:category,
            },
            success: function(response) {
               
                jQuery("#ajax_fitler_search_results").html(response);
                

                console.log(response);
            },
            error: function (error) {
                 console.log(eval(error));
            }     
        });
        e.preventDefault(); 
    });

});