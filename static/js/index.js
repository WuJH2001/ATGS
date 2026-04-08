window.HELP_IMPROVE_VIDEOJS = false;

var INTERP_BASE = "./static/interpolation/stacked";
var NUM_INTERP_FRAMES = 240;

var interp_images = [];
function preloadInterpolationImages() {
  for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path = INTERP_BASE + '/' + String(i).padStart(6, '0') + '.jpg';
    interp_images[i] = new Image();
    interp_images[i].src = path;
  }
}

function setInterpolationImage(i) {
  var image = interp_images[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#interpolation-image-wrapper').empty().append(image);
}


$(document).ready(function() {
    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");

    });

    var options = {
			slidesToScroll: 1,
			slidesToShow: 3,
			loop: true,
			infinite: true,
			autoplay: false,
			autoplaySpeed: 3000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    // Loop on each carousel initialized
    for(var i = 0; i < carousels.length; i++) {
    	// Add listener to  event
    	carousels[i].on('before:show', state => {
    		console.log(state);
    	});
    }

    // Access to bulmaCarousel instance of an element
    var element = document.querySelector('#my-element');
    if (element && element.bulmaCarousel) {
    	// bulmaCarousel instance is available as element.bulmaCarousel
    	element.bulmaCarousel.on('before-show', function(state) {
    		console.log(state);
    	});
    }

    /*var player = document.getElementById('interpolation-video');
    player.addEventListener('loadedmetadata', function() {
      $('#interpolation-slider').on('input', function(event) {
        console.log(this.value, player.duration);
        player.currentTime = player.duration / 100 * this.value;
      })
    }, false);*/
    preloadInterpolationImages();

    $('#interpolation-slider').on('input', function(event) {
      setInterpolationImage(this.value);
    });
    setInterpolationImage(0);
    $('#interpolation-slider').prop('max', NUM_INTERP_FRAMES - 1);

    bulmaSlider.attach();

})


function slide_num(btn, num) {
    const container = btn.closest(".slider-container");
    const slider = container.querySelector(".video-slider");
    const slides = slider.children;
    const total = slides.length;

    // 当前 index（存在 slider 自己身上）
    let currentIndex = slider.dataset.index
        ? parseInt(slider.dataset.index)
        : 0;

    // 暂停当前视频
    const currentVideo = slides[currentIndex].querySelector("video");
    if (currentVideo) currentVideo.pause();

    // 更新 index（循环）
    currentIndex = num;
    if (currentIndex < 0) currentIndex = total - 1;
    if (currentIndex >= total) currentIndex = 0;

    // 滚动
    const slideStyle = getComputedStyle(slides[0]);
    const slideWidth =
        slides[0].offsetWidth + parseInt(slideStyle.marginRight);

    slider.scrollTo({
        left: currentIndex * slideWidth,
        behavior: "smooth"
    });

    // 自动播放新视频（可选）
    const nextVideo = slides[currentIndex].querySelector("video");
    if (nextVideo) nextVideo.play();

    // 保存 index
    slider.dataset.index = currentIndex;
}




function slide(btn, direction) {
    const container = btn.closest(".slider-container");
    const slider = container.querySelector(".video-slider");
    const slides = slider.children;
    const total = slides.length;

    // 当前 index（存在 slider 自己身上）
    let currentIndex = slider.dataset.index
        ? parseInt(slider.dataset.index)
        : 0;

    // 暂停当前视频
    const currentVideo = slides[currentIndex].querySelector("video");
    if (currentVideo) currentVideo.pause();

    // 更新 index（循环）
    currentIndex += direction;
    if (currentIndex < 0) currentIndex = total - 1;
    if (currentIndex >= total) currentIndex = 0;

    // 滚动
    const slideStyle = getComputedStyle(slides[0]);
    const slideWidth =
        slides[0].offsetWidth + parseInt(slideStyle.marginRight);

    slider.scrollTo({
        left: currentIndex * slideWidth,
        behavior: "smooth"
    });

    // 自动播放新视频（可选）
    const nextVideo = slides[currentIndex].querySelector("video");
    if (nextVideo) nextVideo.play();

    // 保存 index
    slider.dataset.index = currentIndex;
}


