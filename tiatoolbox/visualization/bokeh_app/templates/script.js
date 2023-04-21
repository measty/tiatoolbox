const closePopupBtn = document.getElementById('closePopup');
const popup = document.getElementById('popup');
const popupContent = document.querySelector('.popup-content');
const popupHeader = document.querySelector('.popup-header');

let isDragging = false;
let offsetX, offsetY;

closePopupBtn.addEventListener('click', function() {
    popupContent.classList.add('hidden');
});

popupHeader.addEventListener('mousedown', function(e) {
    isDragging = true;
    offsetX = e.clientX - popupContent.getBoundingClientRect().left;
    offsetY = e.clientY - popupContent.getBoundingClientRect().top;
});

document.addEventListener('mousemove', function(e) {
    if (isDragging) {
        popupContent.style.left = (e.clientX - offsetX) + 'px';
        popupContent.style.top = (e.clientY - offsetY) + 'px';
        popupContent.style.position = 'fixed';
    }
});

document.addEventListener('mouseup', function() {
    isDragging = false;
});
