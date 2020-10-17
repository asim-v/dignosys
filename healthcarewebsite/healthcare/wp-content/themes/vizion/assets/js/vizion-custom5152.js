/*
Template: Vizion - AI Startups Responsive WordPress Theme
Author: iqonicthemes.in
Version: 1.0
Design and Developed by: iqonicthemes.in
*/

/*----------------------------------------------
Index Of Script
------------------------------------------------

1.Page Loader
2.Back To Top
3.Background Overlay
3.Tooltip
4.Accordion
5.Header
6.Magnific Popup
7.Countdown
8.counter
9.Owl Carousel
10.Wow Animation
11.Tab Features
12.Contact From


------------------------------------------------
Index Of Script
----------------------------------------------*/
(function($) {

    "use strict";
    jQuery(document).ready(function() {
            jQuery(window).on('load', function(e) {


                /*------------------------
                Page Loader
                --------------------------*/
                jQuery("#load").fadeOut();
                jQuery("#loading").delay(0).fadeOut("slow");

                jQuery(".navbar a").on("click", function(event) {
                    if (!jQuery(event.target).closest(".nav-item.dropdown").length) {
                        jQuery(".navbar-collapse").collapse('hide');
                    }
                });

                /*------------------------
                Progress Bar
                --------------------------*/
                jQuery('.iq-progress-bar > span').each(function() {
                    var jQuerythis = jQuery(this);
                    var width = jQuery(this).data('percent');
                    jQuerythis.css({
                        'transition': 'width 2s'
                    });
                    setTimeout(function() {
                        jQuerythis.appear(function() {
                            jQuerythis.css('width', width + '%');
                        });
                    }, 500);
                });

                /*------------------------
                Back To Top
                --------------------------*/
                jQuery('#back-to-top').fadeOut();
                jQuery(window).on("scroll", function() {
                    if (jQuery(this).scrollTop() > 250) {
                        jQuery('#back-to-top').fadeIn(1400);
                    } else {
                        jQuery('#back-to-top').fadeOut(400);
                    }
                });

                // scroll body to 0px on click
                jQuery('#top').on('click', function() {
                    jQuery('top').tooltip('hide');
                    jQuery('body,html').animate({
                        scrollTop: 0
                    }, 800);
                    return false;
                });

                  /*------------------------
        3 Parallax
        --------------------------*/
     if(jQuery("#scene1").length){
            var scene1 = document.getElementById('scene1');

            var parallaxInstance = new Parallax(scene1);
             }

              if(jQuery("#scene2").length){
            var scene2 = document.getElementById('scene2');

            var parallaxInstance = new Parallax(scene2);
             }
               if(jQuery("#scene3").length){
            var scene3 = document.getElementById('scene3');

            var parallaxInstance = new Parallax(scene3);
             }




                /*------------------------
                Slick Slider
                --------------------------*/
                jQuery(".slick-slider").slick({
                    dots: false,
                    arrows: false,
                    infinite: true,
                    autoplay:true,
                    prevArrow: false,
                    nextArrow: false,
                    centerMode: true,
                    slidesToShow: 3,
                    slidesToScroll: 3,
                    responsive: [
                        {
                            breakpoint: 1024,
                            settings: {
                                slidesToShow: 2,
                                slidesToScroll: 1
                            }
                        },
                        {
                            breakpoint: 980,
                            settings: {
                                slidesToShow: 1,
                                slidesToScroll: 1
                            }
                        }
                    ]
                });

                /*----------------------------
                Slick Slider Car Testimoniol
                ------------------------------*/

                jQuery('.slider-car-for').slick({
                    slidesToShow: 1,
                    slidesToScroll: 1,
                    arrows: true,
                    fade: true,
                    focusOnSelect: true,
                    asNavFor: '.slider-car-nav'
                });
                jQuery('.slider-car-nav').slick({
                    slidesToShow: 3,
                    slidesToScroll: 1,
                    asNavFor: '.slider-car-for',
                    dots: false,
                    centerMode: true,
                    focusOnSelect: true,
                    responsive: [
                        
                        {
                            breakpoint: 767,
                            settings: {
                                slidesToShow: 1,
                                slidesToScroll: 1
                            }
                        }
                    ]
                });

                 jQuery('.verticle-slider').slick({
                    slidesToShow: 3,
                    slidesToScroll: 1,
                    arrows: false,
                     vertical: true,
                     verticalSwiping: true,
                    focusOnSelect: true,
                    autoplay:true,
                   
    centerPadding: '5%',
     onInit: function () {
        jQuery('.slick-active').prev().addClass('prev');
        jQuery('.slick-active').next().addClass('next');
    },

    onBeforeChange: function () {
        jQuery('.slick-slide').removeClass('prev next');
    },

    onAfterChange: function () {
        jQuery('.slick-active').prev().addClass('prev');
        jQuery('.slick-active').next().addClass('next');
    }
                });

                /*----------------------------
                Slick Slider Iq Feedback Slider Testimoniol
                ------------------------------*/

                jQuery('.iq-feedback-slider-for').slick({
                    slidesToShow: 1,
                    slidesToScroll: 1,
                    arrows: false,
                    fade: true,
                    focusOnSelect: true,
                    asNavFor: '.slider-feedback-nav'
                });
                jQuery('.slider-feedback-nav').slick({
                    slidesToShow: 1 ,
                    slidesToScroll: 1,
                    asNavFor: '.iq-feedback-slider-for',
                    dots: true,
                    fade: true,
                    centerMode: false,
                    focusOnSelect: true
                });



                /*------------------------
                5 slider slick computer vizion
                --------------------------*/
                  $('.slider-for').slick({
                  slidesToShow: 1,
                  slidesToScroll: 1,
                  arrows: false,
                  prevArrow: false,
                  nextArrow: false,
                  fade: true,
                  asNavFor: '.slider-nav'
                });
                $('.slider-nav').slick({
                  slidesToShow: 8,
                  slidesToScroll: 1,
                  asNavFor: '.slider-for',
                  dots: false,
                  centerMode: true,
                  focusOnSelect: true,
                  responsive: [{
                    breakpoint: 1024,
                    settings: {
                        slidesToShow: 5,
                        slidesToScroll: 5,
                    }
                }, {
                    breakpoint: 640,
                    settings: {
                        slidesToShow: 4,
                        slidesToScroll: 4,
                    }
                }, {
                    breakpoint: 420,
                    settings: {
                        slidesToShow: 2,
                        slidesToScroll: 2,
                }
                }]

                });


                /*************************
                        Background Overlay
                    *************************/
                jQuery.each(jQuery('.iq-background-overlay'), function() {
                    var $i;
                    var $p;
                    $i = jQuery(this).attr('id');
                    $i = '#' + $i;
                    $p = jQuery($i).parent().parent().parent().addClass($i);
                    jQuery($i).parent().parent().parent().addClass('section_overlay');
                    jQuery($i).insertBefore($p);
                });

                jQuery.each(jQuery('.layer_wrap'), function() {
                    var $i;
                    var $p;
                    $i = jQuery(this).attr('id');
                    $i = '#' + $i;
                    $p = jQuery($i).parent().parent().parent().addClass($i);
                    jQuery($i).parent().parent().parent().addClass('section_overlay');
                    jQuery($i).insertBefore($p);
                });

                /*------------------------
                Tooltip
                --------------------------*/

                jQuery(function() {
                    jQuery('[data-toggle="tooltip"]').tooltip()
                });

                /*------------------------
                Accordion
                --------------------------*/
                jQuery('.iq-accordion .iq-ad-block .ad-details').hide();
                jQuery('.iq-accordion .iq-ad-block:first').addClass('ad-active').children().slideDown('slow');
                jQuery('.iq-accordion .iq-ad-block').on("click", function() {
                    if (jQuery(this).children('div').is(':hidden')) {
                        jQuery('.iq-accordion .iq-ad-block').removeClass('ad-active').children('div').slideUp('slow');
                        jQuery(this).toggleClass('ad-active').children('div').slideDown('slow');
                    }
                });

                jQuery('.iq-accordion .iq-img-block .img-details').hide();
                jQuery('.iq-accordion .iq-img-block:first').addClass('img-active').children().slideDown('slow');
                jQuery('.iq-accordion .iq-img-block').on("click", function() {
                    if (jQuery(this).children('div').is(':hidden')) {
                        jQuery('.iq-accordion .iq-img-block').removeClass('img-active').children('div').slideUp('slow');
                        jQuery(this).toggleClass('img-active').children('div').slideDown('slow');
                    }
                });

                jQuery('.iq-img-faq').hide();

                jQuery('.iq-accordion .iq-ad-block').on("click", function() {
                   // alert(jQuery(this).id());

                   var id = jQuery(this).attr('id');
                   jQuery('#iq-img-'+id).show('slow');
                });


                /*------------------------
                Header
                --------------------------*/

                jQuery(window).on('scroll', function() {
                    if (jQuery(this).scrollTop() > 10) {
                        jQuery('header').addClass('menu-sticky');
                    } else {
                        jQuery('header').removeClass('menu-sticky');
                    }
                });

                /*------------------------
                Magnific Popup
                --------------------------*/

                jQuery('.popup-gallery').magnificPopup({
                    delegate: 'a.popup-img',
                    type: 'image',
                    tLoading: 'Loading image #%curr%...',
                    mainClass: 'mfp-img-mobile',
                    gallery: {
                        enabled: true,
                        navigateByImgClick: true,
                        preload: [0, 1] // Will preload 0 - before current, and 1 after the current image
                    },
                    image: {
                        tError: '<a href="%url%">The image #%curr%</a> could not be loaded.',
                        titleSrc: function(item) {
                            return item.el.attr('title') + '<small>by Marsel Van Oosten</small>';
                        }
                    }
                });


                jQuery('.popup-youtube, .popup-vimeo, .popup-gmaps').magnificPopup({
                    disableOn: 700,
                    type: 'iframe',
                    mainClass: 'mfp-fade',
                    removalDelay: 160,
                    preloader: false,
                    fixedContentPos: false
                });


                /*------------------------
                Countdown
                --------------------------*/
                jQuery('#countdown').countdown({
                    date: '10/01/2019 23:59:59',
                    day: 'Day',
                    days: 'Days'
                });

                /*------------------------
                counter
                --------------------------*/
                jQuery('.timer').countTo();


                /*------------------------
        8 Auto tab
        --------------------------*/

            var HM = {
    //tab
    jqs_slideList: '.slideList',
    jqs_tabList: '.slides .carouselLinks',


    init: function() {
        //init sliders
        var aSliders = $(this.jqs_slideList);
        if (aSliders.length > 0) {
            this.slideShow(aSliders);
        }

        //init the carousels that are lists of links
        // $('.carousel.icons').hellmannsCrsl({
        //     rotateSpeed: 5000,
        //     viewport: '.carouselLinks'
        // });
    },

    slideShow: function(eSlideListParam) {
        var slideList = eSlideListParam,
            slides = slideList.find('li'),
            tabList = slideList.siblings('.carouselLinks'),
            tabs = tabList.find('.object-new'),
            speed = 500;


        tabs.on('click', 'a', function(e) {
            $(this).trigger('slides.swap');
            e.preventDefault();
        });

        //make it automatic, but this doesn't work properly, I'm stuck...
        setInterval(function() {
            var current = parseInt($('li.selected a').data('links-to').split('_')[1],10);
            var idx=current-1;
            var max = $('.carouselLinks li a').length;
            idx = (current<max) ? (idx+1):0;
            $('.object-new a:eq('+idx+')').trigger('click');
        }, 3000);

        /**
         * This is where the animation, i.e. fade, is performing.
         * I find it quite convenient to use bind/trigger principle as it's easier to maintain
         */
        tabs.find('a').bind('slides.swap', function() {
            var self = $(this),
                selfIndex = self.parent().index(),
                targetSlide = slides.eq(selfIndex);

            //fade in/out slides
            slides.filter('.active').stop(true, false).fadeOut(speed, function() {
                $(this).removeClass('active');
            });
            targetSlide.stop(true, false).fadeIn(speed).addClass('active');

            tabs.removeClass('selected');
            self.parent().addClass('selected');
        });
    }
};

HM.init();

                /*------------------------
                Owl Carousel
                --------------------------*/
                jQuery('.owl-carousel').each(function() {
                    var $carousel = jQuery(this);
                    $carousel.owlCarousel({
                        items: $carousel.data("items"),
                        loop: $carousel.data("loop"),
                        margin: $carousel.data("margin"),
                        nav: $carousel.data("nav"),
                        dots: $carousel.data("dots"),
                        autoplay: $carousel.data("autoplay"),
                        autoplayTimeout: $carousel.data("autoplay-timeout"),
                        navText: ["<i class='fa fa-angle-left fa-2x'></i>", "<i class='fa fa-angle-right fa-2x'></i>"],
                        responsiveClass: true,
                        responsive: {
                            // breakpoint from 0 up
                            0: {
                                items: $carousel.data("items-mobile-sm"),
                                nav: false,
                                dots: true
                            },
                            // breakpoint from 480 up
                            480: {
                                items: $carousel.data("items-mobile"),
                                nav: false,
                                dots: true
                            },
                            // breakpoint from 786 up
                            786: {
                                items: $carousel.data("items-tab")
                            },
                            // breakpoint from 1023 up
                            1023: {
                                items: $carousel.data("items-laptop")
                            },
                            1199: {
                                items: $carousel.data("items")
                            }
                        }
                    });
                });

                /*------------------------
                Wow Animation
                --------------------------*/
                var wow = new WOW({
                    boxClass: 'wow',
                    animateClass: 'animated',
                    offset: 0,
                    mobile: false,
                    live: true
                });
                wow.init();


                /*-------------------------------------
                ------------------------------------*/

                /*------------------------
                 Tool tip
                --------------------------*/

                // set the wrapper width and height to match the img size
                jQuery('#iq-tooltip').css({'width':jQuery('#iq-tooltip img').width(),
                    'height':jQuery('#iq-tooltip img').height()
                })
                
                //tooltip direction
                var tooltipDirection;
                             
                for (i=0; i<jQuery(".pin").length; i++) {               
                    // set tooltip direction type - up or down             
                    if (jQuery(".pin").eq(i).hasClass('pin-down')) {
                        tooltipDirection = 'tooltip-down  tooltip-down-'+i;
                    } else {
                        tooltipDirection = 'tooltip-up';
                    }
                    // append the tooltip
                    jQuery("#iq-tooltip").append(
                        "<div style='left:"+jQuery(".pin").eq(i).data('xpos')+"px;top:"+jQuery(".pin").eq(i).data('ypos')+"px' class='" + tooltipDirection +"'>\
                                <div class='tooltip'>" + jQuery(".pin").eq(i).html() + "</div>\
                        </div>"
                    );
                }    
                
                // show/hide the tooltip
                jQuery('.tooltip-up, .tooltip-down').mouseenter(function() {
                            jQuery(this).children('.tooltip').fadeIn(100);
                        }).mouseleave(function(){
                            jQuery(this).children('.tooltip').fadeOut(100);
                })  

    
                /*------------------------
                    Vertical slider
                --------------------------*/
                
                // Get titles from the DOM
                var titleMain  = jQuery("#animatedHeading");
                var titleSubs  = titleMain.find("slick-active");

                if (titleMain.length) {

                  titleMain.slick({
                    autoplay: false,
                    arrows: false,
                    dots: false,
                    slidesToShow: 3,
                    centerPadding: "10px",
                    draggable: false,
                    infinite: true,
                    pauseOnHover: false,
                    swipe: true,
                    touchMove: true,
                    vertical: true,
                    verticalScrolling: true,
                    speed: 1000,
                    autoplaySpeed: 2000,
                    useTransform: true,
                    cssEase: 'cubic-bezier(0.645, 0.045, 0.355, 1.000)',
                    adaptiveHeight: true
                  });

                  // On init
                  jQuery(".slick-dupe").each(function(index, el) {
                    jQuery("#animatedHeading").slick('slickAdd', "<div>" + el.innerHTML + "</div>");
                  });

                  // Manually refresh positioning of slick
                  titleMain.slick('slickPlay');
                };        


                /*------------------------
                Tab Features
                --------------------------*/
                jQuery('#myTab li a').on('click', function() {
                    jQuery('#myTab li a').attr('aria-selected', false);
                    jQuery(this).attr('aria-selected', true);
                });

                jQuery(window).on('scroll', function(e) {

                    var nav = jQuery('#features');
                    if (nav.length) {
                        var contentNav = nav.offset().top - 250;
                        if (jQuery(this).scrollTop() >= (contentNav)) {
                            e.preventDefault();
                            jQuery('#features .row #myTab li a').removeClass('active');
                            jQuery('#features .row #myTab li').children('a[aria-selected=true]').addClass('active');
                        }
                    }
                });

                jQuery('#iqTab li a').on('click', function() {
                    jQuery('#iqTab li a').attr('aria-selected', false);
                    jQuery(this).attr('aria-selected', true);
                });

                jQuery(window).on('scroll', function(e) {

                    var nav = jQuery('#features2');
                    if (nav.length) {
                        var contentNav = nav.offset().top - 250;
                        if (jQuery(this).scrollTop() >= (contentNav)) {
                            e.preventDefault();
                            jQuery('#features2 .row #iqTab li a').removeClass('active');
                            jQuery('#features2 .row #iqTab li').children('a[aria-selected=true]').addClass('active');
                        }
                    }
                });

                /*---------------------------
                    Tabs
                ---------------------------*/
                jQuery(document).ready(function(){
                    var a=jQuery('.nav-pills').each(function(){
                        var b =jQuery(this).find('a.active').attr('href');
                        activaTab(b);
                    })
                });

                function activaTab(pill){
                    jQuery(pill).addClass('active show');
                };


                jQuery('.sub-menu').css('display', 'none');
                jQuery('.sub-menu').prev().addClass('isubmenu');
                jQuery(".sub-menu").before('<i class="fa fa-angle-down toggledrop" aria-hidden="true"></i>');


                jQuery('.widget .fa.fa-angle-down, #main .fa.fa-angle-down').on('click', function() {
                    jQuery(this).next('.children, .sub-menu').slideToggle();
                });

                jQuery("#top-menu .menu-item .toggledrop").off("click");
                if (jQuery(window).width() < 992) {
                    jQuery('#top-menu .menu-item .toggledrop').on('click', function(e) {
                        e.preventDefault();
                        jQuery(this).next('.children, .sub-menu').slideToggle();
                    });
                }

            });

            jQuery(window).on('resize', function() { "use strict";
            jQuery('.widget .fa.fa-angle-down, #main .fa.fa-angle-down').on('click', function() {
                jQuery(this).next('.children, .sub-menu').slideToggle();
            });

            jQuery("#top-menu .menu-item .toggledrop").off("click");
            if (jQuery(window).width() < 992) {
                jQuery('#top-menu .menu-item .toggledrop').on('click', function(e) {
                    e.preventDefault();
                    jQuery(this).next('.children, .sub-menu').slideToggle();
                });
            }
        });
    });

})(jQuery);
