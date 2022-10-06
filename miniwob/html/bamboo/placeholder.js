//https://github.com/parndt/jquery-html5-placeholder-shim/blob/master/jquery.html5-placeholder-shim.js
(function($) {
	// @todo Document this.
	$.extend($, {placeholder: {
			browser_supported: function() {
				return this._supported !== undefined ?
						this._supported :
						(this._supported = !!('placeholder' in $('<input type="text">')[0]));
			},
			shim: function(opts) {
				var config = {
					color: '#999',
					cls: 'placeholder noclick',
					selector: 'input[placeholder]:not(".js-placeholderShimIgnore"), textarea[placeholder]:not(".js-placeholderShimIgnore")'
				};
				$.extend(config, opts);
				return $(config.selector)._placeholder_shim(config);
			},
			check: function(item) { //bhr-jn added to check if datefield has value on close of calendar popup and hide placeholder if so
				return $(item)._placeholder_check(item);
			}
		}});

	$.extend($.fn, {
		_placeholder_shim: function(config) {
			var jadeIsEnabled = window.BambooHR.FeatureToggle.isEnabled('jade');

			function calcPositionCss(target) {
				var $target = $(target);
				var $op = $target.offsetParent().offset();
				var $ot = $target.offset();
				var ewidth = $target.width();

				if ($target.hasClass('calendar-field')) {
					// Account for width of calendar field icon
					ewidth = ($target.width() - 23);
				}

				// Round to avoid partial whole numbers
				var tmpTop = (parseInt($target.css('margin-top')) + parseInt($target.css('border-top-width')));

				if ($(target).is('textarea')) {
					tmpTop = (parseInt($target.css('margin-top')) + parseInt($target.css('border-top-width')));
				}

				if (BambooHR.FeatureToggle.isEnabled('jade')) {
					ewidth = $target.outerWidth();
					tmpTop = 0;
				}

				return {
					top: tmpTop + "px",
					left: ($ot.left - $op.left) + ($target.outerWidth() - $target.innerWidth())/2,
					width: ewidth
				};
			}

			function adjustToResizing(label) {
				var $target = label.data('target');
				if (typeof $target !== "undefined") {
					label.css(calcPositionCss($target));
					$(window).one("resize", function() {
						adjustToResizing(label);
					});
				}
			}
			return this.each(function() {
				var $this = $(this);
				var placeholder = $this.attr('placeholder');

				if($.placeholder.browser_supported()
					&& (
						$this.data('placeholderTrigger') != 'keydown'
						&& $this.data('placeholderTrigger') != 'focus'
						&& !$this.closest('form').hasClass(config.autoFormClass)
					)
				) {
					return true;
				}
				if ($this.is(':visible')) {

					if ($this.data('placeholder2')) {
						var $ol = $this.data('placeholder2');
						$ol.css(calcPositionCss($this));

						if (placeholder) {
							$ol.text(placeholder);
							$this.attr('placeholder', '');
						}

						$.placeholder.check($this);

						return true;
					}

					var possible_line_height = {};
					if (!$this.is('textarea') && $this.css('height') != 'auto') {

						if ($this.parent().hasClass('inputIcons')) {
							tHeight = (parseInt($this.css('height')) - 2) + 'px';
						} else {
							tHeight = $this.css('height');
						}

						if (parseInt($this.css('line-height')) > 0 && $this.parent().hasClass('inputIcons')) {
							possible_line_height = {lineHeight: (parseInt($this.css('line-height')) + 2) + 'px', whiteSpace: 'nowrap'};
						} else if (parseInt($this.css('line-height')) > 0 && !$this.parent().hasClass('inputIcons')) {
							possible_line_height = {lineHeight: $this.css('line-height'), whiteSpace: 'nowrap'};
						} else {
							possible_line_height = {lineHeight: tHeight, whiteSpace: 'nowrap'};
						}

						if (window.BambooHR.FeatureToggle.isEnabled('jade')) {
							possible_line_height = {lineHeight: '1', whiteSpace: 'nowrap'};
						}

					} else if ($this.is('textarea')) {
						possible_line_height = {lineHeight: $this.css('line-height')};
					}

					var display = 'inline';

					if (
						jadeIsEnabled &&
						!$this.is('textarea')
					) {
						display = 'flex';
					}

					var ol = $('<label />')
							.text($this.attr('placeholder'))
							.addClass(config.cls)
							.css($.extend({
						alignItems: 'center',
						position: 'absolute',
						display: display,
						'float': 'none',
						overflow: 'hidden',
						textAlign: 'left',
						color: config.color,
						cursor: 'text',
						paddingTop: (parseInt($this.css('padding-top')) + parseInt($this.css('border-top-width'))) + "px",
						paddingRight: $this.css('padding-right'),
						paddingBottom: $this.css('padding-bottom'),
						paddingLeft: $this.css('padding-left'),
						fontSize: $this.css('font-size'),
						fontFamily: $this.css('font-family'),
						fontStyle: $this.css('font-style'),
						fontWeight: $this.css('font-weight'),
						height: $this.css('height'),
						textTransform: $this.css('text-transform'),
						backgroundColor: 'transparent',
						boxSizing: 'border-box',
						zIndex: 1
					}, possible_line_height))
						.css(calcPositionCss(this))
						.attr('for', this.id)
						.data('target', $this)
						.click(function() {
							$(this).data('target').focus();
						})
						.insertBefore(this);
					$this.data('placeholder2', ol);
					$this.attr('placeholder','');

					$.placeholder.check($this);

					if($this.data('placeholderTrigger') == 'focus') {
						$this
							.on('paste', function() {
								ol.hide();
							})
							.keydown(function() {	// Required for when autofocus is used
								if (jadeIsEnabled) {
									ol.hide();
								}
							})
							.focus(function() {
								ol.hide();
							})
							.blur(function() {
								if ($this.get(0).tagName == "DIV")
									ol[$this.text().length ? 'hide' : 'show']();
								else
									ol[$this.val().length ? 'hide' : 'show']();
								//jn - show placeholder text on blur if no user selected value present
								//		 check last character isn't last character in mask (partial fill)
								var cMask = window.GLOBAL_DATE_MASK || 'mm/dd/yyyy';

								if ($this.val().slice(-1) == cMask.slice(-1) && !jadeIsEnabled) {
									$this.prev('label.placeholder').show();
								}
							});
					} else {
						$this
							.on('paste', function() {
								ol.hide();
							})
							.keydown(function() {
								setTimeout(function() { //timeout necessary because .val() doesn't exist at instant of keydown
									if ($this.get(0).tagName == "DIV")
										ol[$this.text().length ? 'hide' : 'show']();
									else
										ol[$this.val().length ? 'hide' : 'show']();
								});
							}).blur(function() {
								var cMask = window.GLOBAL_DATE_MASK || 'mm/dd/yyyy';
								if ($this.get(0).tagName == "DIV")
									ol[$this.text().length && $this.text() !== cMask ? 'hide' : 'show']();
								else
									ol[$this.val().length && $this.val() !== cMask ? 'hide' : 'show']();
							}).triggerHandler('blur');
					}
					$(window).one("resize", function() {
						adjustToResizing(ol);
					});
				}
			});
		},

		_placeholder_check: function (item) {
			var $placeholder = item.prev('label.placeholder');
			if (item.val().length) {
				$placeholder.hide();
			} else {
				$placeholder.show();
			}
		}
	});
})(jQuery);
