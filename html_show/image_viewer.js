/* Same-page viewer for dynamically rendered tool images. */
(function () {
  'use strict';
  var modal, image, backdrop, viewport, status, zoomIn, zoomOut, scaleLabel, closeButton;
  var opener = null, scale = 1, fitScale = 1, minScale = 0.1, maxScale = 4;
  var ready = false, requestId = 0, previousBodyOverflow, previousRootOverflow;

  function createViewer() {
    if (modal) return;
    modal = document.createElement('div');
    modal.id = 'evalImageViewer';
    modal.className = 'eval-image-modal';
    modal.hidden = true;
    modal.setAttribute('role', 'dialog');
    modal.setAttribute('aria-modal', 'true');
    modal.setAttribute('aria-label', 'Image preview');
    modal.innerHTML = '<img class="eval-image-modal-backdrop" alt="" aria-hidden="true">' +
      '<div class="eval-image-toolbar" role="toolbar" aria-label="Image controls">' +
      '<span class="eval-image-scale" aria-live="polite"></span>' +
      '<button type="button" data-image-action="zoom-in" aria-label="Zoom in" title="Zoom in (+)"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><circle cx="12" cy="12" r="9.5"/><path d="M7 12h10M12 7v10"/></svg></button>' +
      '<button type="button" data-image-action="zoom-out" aria-label="Zoom out" title="Zoom out (-)"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><circle cx="12" cy="12" r="9.5"/><path d="M7 12h10"/></svg></button>' +
      '<button type="button" data-image-action="close" aria-label="Close image" title="Close (Esc)">❎</button>' +
      '</div><div class="eval-image-viewport" tabindex="0" aria-label="Image; scroll to inspect when zoomed in">' +
      '<div class="eval-image-canvas"><img class="eval-image-full" alt="Tool image" hidden draggable="false"></div></div>' +
      '<p class="eval-image-status" role="status"></p>';
    document.body.appendChild(modal);
    image = modal.querySelector('.eval-image-full');
    backdrop = modal.querySelector('.eval-image-modal-backdrop');
    viewport = modal.querySelector('.eval-image-viewport');
    status = modal.querySelector('.eval-image-status');
    scaleLabel = modal.querySelector('.eval-image-scale');
    zoomIn = modal.querySelector('[data-image-action="zoom-in"]');
    zoomOut = modal.querySelector('[data-image-action="zoom-out"]');
    closeButton = modal.querySelector('[data-image-action="close"]');
    zoomIn.addEventListener('click', function () { setScale(scale * 1.25); });
    zoomOut.addEventListener('click', function () { setScale(scale / 1.25); });
    closeButton.addEventListener('click', closeViewer);
    modal.addEventListener('click', function (event) {
      if (event.target !== image && !event.target.closest('.eval-image-toolbar')) closeViewer();
    });
  }

  function fittedScale() {
    var style = window.getComputedStyle(viewport);
    var width = viewport.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight);
    var height = viewport.clientHeight - parseFloat(style.paddingTop) - parseFloat(style.paddingBottom);
    return Math.min(1, Math.max(1, width) / image.naturalWidth, Math.max(1, height) / image.naturalHeight);
  }

  function setScale(next) {
    if (!ready) return;
    scale = Math.max(minScale, Math.min(maxScale, next));
    image.style.width = (image.naturalWidth * scale) + 'px';
    image.style.height = (image.naturalHeight * scale) + 'px';
    scaleLabel.textContent = Math.round(scale * 100) + '%';
    zoomIn.disabled = scale >= maxScale;
    zoomOut.disabled = scale <= minScale;
    // Keep the enlarged image centered, with every edge reachable by scrolling.
    viewport.scrollLeft = Math.max(0, (viewport.scrollWidth - viewport.clientWidth) / 2);
    viewport.scrollTop = Math.max(0, (viewport.scrollHeight - viewport.clientHeight) / 2);
  }

  function openViewer(button) {
    createViewer();
    var url = button.getAttribute('data-eval-image');
    if (!url || button.disabled) return;
    if (modal.hidden) {
      previousBodyOverflow = document.body.style.overflow;
      previousRootOverflow = document.documentElement.style.overflow;
    }
    opener = button;
    ready = false;
    var currentRequest = ++requestId;
    image.hidden = true;
    image.style.width = '';
    image.style.height = '';
    status.textContent = 'Loading image…';
    scaleLabel.textContent = '';
    zoomIn.disabled = zoomOut.disabled = true;
    modal.hidden = false;
    document.body.style.overflow = 'hidden';
    document.documentElement.style.overflow = 'hidden';
    image.onload = function () {
      if (currentRequest !== requestId || modal.hidden || !image.naturalWidth) return;
      ready = true;
      status.textContent = '';
      image.hidden = false;
      fitScale = fittedScale();
      minScale = Math.min(0.1, fitScale);
      setScale(fitScale);
    };
    image.onerror = function () {
      if (currentRequest !== requestId || modal.hidden) return;
      status.textContent = 'Unable to load this image.';
      ready = false;
      image.hidden = true;
      zoomIn.disabled = zoomOut.disabled = true;
    };
    var thumbnail = button.querySelector('.eval-image-thumbnail');
    image.alt = thumbnail ? thumbnail.alt : 'Tool image';
    image.src = url;
    backdrop.src = url;
    closeButton.focus({preventScroll: true});
  }

  function closeViewer() {
    if (!modal || modal.hidden) return;
    ++requestId;
    ready = false;
    modal.hidden = true;
    image.onload = image.onerror = null;
    image.removeAttribute('src');
    backdrop.removeAttribute('src');
    document.body.style.overflow = previousBodyOverflow;
    document.documentElement.style.overflow = previousRootOverflow;
    if (opener && opener.isConnected) opener.focus({preventScroll: true});
    opener = null;
  }

  document.addEventListener('click', function (event) {
    var button = event.target.closest('.eval-image-preview');
    if (!button) return;
    event.preventDefault();
    event.stopPropagation();
    openViewer(button);
  });
  document.addEventListener('error', function (event) {
    if (!event.target.classList || !event.target.classList.contains('eval-image-thumbnail')) return;
    var button = event.target.closest('.eval-image-preview');
    if (button) {
      button.classList.add('is-unavailable');
      button.disabled = true;
    }
  }, true);
  document.addEventListener('keydown', function (event) {
    if (!modal || modal.hidden) return;
    if (event.key === 'Escape') {
      event.preventDefault();
      event.stopImmediatePropagation();
      closeViewer();
    } else if (!event.ctrlKey && !event.metaKey && (event.key === '+' || event.key === '=' || event.key === '-')) {
      event.preventDefault();
      setScale(event.key === '-' ? scale / 1.25 : scale * 1.25);
    } else if (event.key === 'Tab') {
      var controls = [zoomIn, zoomOut, closeButton, viewport].filter(function (el) { return !el.disabled; });
      var index = controls.indexOf(document.activeElement);
      if (index < 0 || (!event.shiftKey && index === controls.length - 1) || (event.shiftKey && index === 0)) {
        event.preventDefault();
        controls[event.shiftKey ? controls.length - 1 : 0].focus({preventScroll: true});
      }
    }
  }, true);
  window.addEventListener('resize', function () {
    if (!modal || modal.hidden || !ready) return;
    var wasFitted = Math.abs(scale - fitScale) < 0.0001;
    fitScale = fittedScale();
    minScale = Math.min(0.1, fitScale);
    setScale(wasFitted ? fitScale : scale);
  });
}());
