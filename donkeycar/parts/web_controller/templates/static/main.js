var driveHandler = new function() {
    //functions used to drive the vehicle. 

    var state = {
        'tele': {
            "user": {
                'angle': 0,
                'throttle': 0,
            },
            "pilot": {
                'angle': 0,
                'throttle': 0,
            }
        },
        'brakeOn': true,
        'recording': false,
        'driveMode': "user",
        'pilot': 'None',
        'session': 'None',
        'lag': 0,
        'controlMode': 'joystick',
        'maxThrottle' : 1,
        'throttleMode' : 'user',
        'buttons': {
            "w1": false,  // boolean; true is 'down' or pushed, false is 'up' or not pushed
            "w2": false,
            "w3": false,
            "w4": false,
            "w5": false,
        }
    }

    var joystick_options = {}
    var joystickLoopRunning=false;

    var hasGamepad = false;

    var deviceHasOrientation=false;
    var initialGamma;

    var vehicle_id = ""
    var driveURL = ""
    var socket
    var tuningSocket
    // Mirror of LocalWebController.tuning on the server. Populated by
    // /wsTuning snapshot messages; mutated by slider input events.
    var tuningState = {
        hsv_center_low:  [0, 0, 0],
        hsv_center_high: [179, 255, 255],
        hsv_edge_low:    [0, 0, 0],
        hsv_edge_high:   [179, 255, 255],
        pid_p: 0.0, pid_i: 0.0, pid_d: 0.0,
        throttle_min: 0.0, throttle_max: 1.0,
        scan_y: 0, scan_height: 0,
        steering_left_pwm: 0, steering_right_pwm: 0,
        ai_throttle_mult: 1.0,
        ai_steering_mult: 1.0,
        line_follower_mode: 'center_line',
        half_track_width_px: 80,
    };

    this.load = function() {
      driveURL = '/drive'
      socket = new WebSocket('ws://' + location.host + '/wsDrive');
      tuningSocket = new WebSocket('ws://' + location.host + '/wsTuning');
      console.log('[tuning] socket constructed, state=', tuningSocket.readyState);
      tuningSocket.onopen = function() {
          console.log('[tuning] socket OPEN');
      };
      tuningSocket.onerror = function(e) {
          console.error('[tuning] socket ERROR', e);
      };
      tuningSocket.onclose = function(e) {
          console.warn('[tuning] socket CLOSED', e.code, e.reason);
      };
      tuningSocket.onmessage = function(evt) {
          console.log('[tuning] recv:', evt.data.substring(0, 200));
          var msg;
          try { msg = JSON.parse(evt.data); } catch (e) {
              console.error('[tuning] bad JSON', e);
              return;
          }
          if (msg.type === 'snapshot' && msg.tuning) {
              Object.assign(tuningState, msg.tuning);
              console.log('[tuning] snapshot applied, painting UI');
              paintTuningUI();
          } else if (msg.type === 'rejected') {
              console.warn('[tuning] server rejected', msg.rejections);
              flashTuningStatus('rejected: ' + msg.rejections.map(function(r){return r.key;}).join(', '), true);
          }
      };

      setBindings()

      joystick_element = document.getElementById('joystick_container');
      joystick_options = {
        zone: joystick_element,  // active zone
        mode: 'dynamic',
        size: 200,
        color: '#668AED',
        dynamicPage: true,
        follow: true,
      };

      var manager = nipplejs.create(joystick_options);
      bindNipple(manager)

      if(!!navigator.getGamepads){
        console.log("Device has gamepad support.")
        hasGamepad = true;
      }

      if (window.DeviceOrientationEvent) {
        window.addEventListener("deviceorientation", handleOrientation);
        console.log("Browser supports device orientation, setting control mode to tilt.");
        state.controlMode = 'tilt';
        deviceOrientationLoop();
      } else {
        console.log("Device Orientation not supported by browser, setting control mode to joystick.");
        state.controlMode = 'joystick';
      }
    };

    //
    // Update a state object with the given data.
    // This will only update existing fields in 
    // the state; it will not add new fields that
    // may exist in the data but not the state.
    //
    var updateState = function(state, data) {
        let changed = false;
        if(typeof data === 'object') {
            const keys = Object.keys(data)
            keys.forEach(key => {
                //
                // state must already have the key;
                // we are not adding new fields to the state,
                // we are only updating existing fields.
                //
                if(state.hasOwnProperty(key) && state[key] !== data[key]) {
                    if(typeof state[key] === 'object') {
                        // recursively update the state's object field
                        changed = updateState(state[key], data[key]) && changed;
                    } else {
                        state[key] = data[key];
                        changed = true;
                    }
                }
            });
        }
        return changed;
    }

    var setBindings = function() {
      //
      // when server sends a message with state changes
      // then update our local state and 
      // if there were any changes then redraw the UI.
      //
      socket.onmessage = function (event) {
        console.log(event.data);
        const data = JSON.parse(event.data);
        if(updateState(state, data)) {
            updateUI();
        }
      };

      $(document).keydown(function(e) {
          if(e.which == 32) { toggleBrake() }  // 'space'  brake
          if(e.which == 82) { toggleRecording() }  // 'r'  toggle recording
          if(e.which == 73) { throttleUp() }  // 'i'  throttle up
          if(e.which == 75) { throttleDown() } // 'k'  slow down
          if(e.which == 74) { angleLeft() } // 'j' turn left
          if(e.which == 76) { angleRight() } // 'l' turn right
          if(e.which == 65) { updateDriveMode('local') } // 'a' turn on local mode (full _A_uto)
          if(e.which == 85) { updateDriveMode('user') } // 'u' turn on manual mode (_U_user)
          if(e.which == 83) { updateDriveMode('local_angle') } // 's' turn on local mode (auto _S_teering)
          if(e.which == 77) { toggleDriveMode() } // 'm' toggle drive mode (_M_ode)
      });

      $('#mode_select').on('change', function () {
        updateDriveMode($(this).val());
      });

      $('#max_throttle_select').on('change', function () {
        state.maxThrottle = parseFloat($(this).val());
      });

      $('#throttle_mode_select').on('change', function () {
        state.throttleMode = $(this).val();
      });

      $('#record_button').click(function () {
        toggleRecording();
      });

      $('#brake_button').click(function() {
        toggleBrake();
      });

      $('input[type=radio][name=controlMode]').change(function() {
        if (this.value == 'joystick') {
          state.controlMode = "joystick";
          joystickLoopRunning = true;
          console.log('joystick mode');
          joystickLoop();
        } else {
          joystickLoopRunning = false;
        }

        if (deviceHasOrientation && this.value == 'tilt') {
          state.controlMode = "tilt";
          console.log('tilt mode')
        }

        if (hasGamepad && this.value == 'gamepad') {
          state.controlMode = "gamepad";
          console.log('gamepad mode')
          gamePadLoop();
        }
        updateUI();
      });

      // programmable buttons
      $('#button_bar > button').mousedown(function() {
        console.log(`${$(this).attr('id')} mousedown`);
        state.buttons[$(this).attr('id')] = true;
        postDrive(["buttons"]); // write it back to the server
      });
      $('#button_bar > button').mouseup(function() {
        console.log(`${$(this).attr('id')} mouseup`);
        state.buttons[$(this).attr('id')] = false;
        postDrive(["buttons"]); // write it back to the server
      });

      // ====== Tuning panel bindings ======
      bindTuningSliders();
      bindInputModeToggle();
      bindTuningSectionCollapse();
      $('#copy-snippet-btn').on('click', function() {
        fetch('/tuning/snippet')
          .then(function(r) { return r.text(); })
          .then(function(text) {
            // navigator.clipboard.writeText only works in secure
            // contexts (https or http://localhost). On a Pi served
            // over plain http we have to fall back to the legacy
            // document.execCommand('copy') path.
            if (navigator.clipboard && navigator.clipboard.writeText) {
              console.log('[tuning] copy: using navigator.clipboard');
              return navigator.clipboard.writeText(text).then(function() {
                flashTuningStatus('snippet copied');
              }).catch(function(err) {
                console.warn('[tuning] navigator.clipboard failed, falling back', err);
                if (legacyCopyToClipboard(text)) {
                  flashTuningStatus('snippet copied');
                } else {
                  flashTuningStatus('copy failed (see console)', true);
                }
              });
            }
            console.log('[tuning] copy: navigator.clipboard unavailable, using execCommand fallback');
            if (legacyCopyToClipboard(text)) {
              flashTuningStatus('snippet copied');
            } else {
              flashTuningStatus('copy failed (see console)', true);
            }
          })
          .catch(function(err) {
            console.warn('[tuning] snippet fetch failed', err);
            flashTuningStatus('copy failed (see console)', true);
          });
      });

      // Reveal the paste box and focus it so the user can drop in a snippet.
      $('#paste-snippet-btn').on('click', function() {
        $('#paste-snippet-box').show();
        $('#paste-snippet-text').focus();
      });
      $('#cancel-snippet-btn').on('click', function() {
        $('#paste-snippet-text').val('');
        $('#paste-snippet-box').hide();
      });
      // POST the pasted text to /tuning/snippet; the server parses it, applies
      // it through the normal validate/commit path, and broadcasts a snapshot
      // that repaints the UI for us (tuningSocket.onmessage -> paintTuningUI).
      $('#apply-snippet-btn').on('click', function() {
        var text = $('#paste-snippet-text').val();
        if (!text || !text.trim()) {
          flashTuningStatus('nothing to apply', true);
          return;
        }
        fetch('/tuning/snippet', { method: 'POST', body: text })
          .then(function(r) { return r.json(); })
          .then(function(res) {
            var applied = (res.applied || []).length;
            var rejected = (res.rejections || []).length;
            if (applied === 0 && rejected === 0) {
              flashTuningStatus('no recognized keys in snippet', true);
              return;
            }
            var msg = 'applied ' + applied;
            if (rejected) msg += ', rejected ' + rejected + ' (' +
              res.rejections.map(function(r){ return r.key; }).join(', ') + ')';
            flashTuningStatus(msg, rejected > 0);
            $('#paste-snippet-text').val('');
            $('#paste-snippet-box').hide();
          })
          .catch(function(err) {
            console.warn('[tuning] snippet apply failed', err);
            flashTuningStatus('apply failed (see console)', true);
          });
      });
    };

    // Copy `text` to the clipboard via a hidden <textarea> and
    // document.execCommand('copy'). Works on plain http:// where
    // navigator.clipboard is undefined. Returns true on success.
    function legacyCopyToClipboard(text) {
        var ta = document.createElement('textarea');
        ta.value = text;
        ta.readOnly = true;
        ta.style.position = 'fixed';
        ta.style.top = '0';
        ta.style.left = '0';
        ta.style.opacity = '0';
        document.body.appendChild(ta);
        var ok = false;
        try {
            ta.focus();
            ta.select();
            ok = document.execCommand('copy');
            console.log('[tuning] legacyCopyToClipboard execCommand =', ok);
        } catch (e) {
            console.warn('[tuning] legacyCopyToClipboard threw', e);
            ok = false;
        } finally {
            document.body.removeChild(ta);
        }
        return ok;
    }

    function sendTuningPatch(patch) {
        var state = tuningSocket ? tuningSocket.readyState : 'no-socket';
        console.log('[tuning] sendTuningPatch state=', state, 'patch=', patch);
        if (tuningSocket && tuningSocket.readyState === 1) {
            tuningSocket.send(JSON.stringify({ set: patch }));
        } else {
            console.warn('[tuning] DROPPED patch: socket not OPEN (state=' + state + ')');
        }
    }

    // Debounce so dragging a slider doesn't flood the websocket.
    function debounce(fn, ms) {
        var t = null;
        return function() {
            var args = arguments, self = this;
            clearTimeout(t);
            t = setTimeout(function() { fn.apply(self, args); }, ms);
        };
    }

    function flashTuningStatus(text, isError) {
        var $el = $('#tuning-status');
        if (!$el.length) return;
        $el.text(text).css('color', isError ? '#c00' : '#888');
        setTimeout(function() { $el.text(''); }, 2000);
    }

    // Show only the fieldsets whose data-modes attribute lists `mode`.
    // Fieldsets without data-modes (none today) are left untouched.
    function applyModeVisibility(mode) {
        var nodes = document.querySelectorAll('[data-modes]');
        for (var i = 0; i < nodes.length; i++) {
            var el = nodes[i];
            var allowed = (el.dataset.modes || '').split(/\s+/);
            el.style.display = allowed.indexOf(mode) >= 0 ? '' : 'none';
        }
    }

    // Persist each tuning subsection's collapsed/expanded state in localStorage
    // so the user can hide everything except the one knob they're working on,
    // and that layout survives a page reload. localStorage is wrapped in
    // try/catch because Safari private mode and some embedded WebViews throw
    // SecurityError on access — we still want the panel usable, just without
    // persistence. Restore goes through Bootstrap's collapse plugin (not raw
    // class manipulation) so aria-expanded on the trigger is set correctly
    // and the chevron CSS (which keys off aria-expanded) stays in sync.
    function safeStorageGet(key, defaultValue) {
        try {
            var v = localStorage.getItem(key);
            return v !== null ? v : (defaultValue !== undefined ? defaultValue : null);
        }
        catch (e) { return defaultValue !== undefined ? defaultValue : null; }
    }
    function safeStorageSet(key, value) {
        try { localStorage.setItem(key, value); }
        catch (e) { /* private mode / blocked storage — silently no-op */ }
    }

    function bindTuningSectionCollapse() {
        var $sections = $('#tuning-body .panel-collapse[id^="tune-sec-"]');
        $sections.each(function() {
            var saved = safeStorageGet('tuneCollapse:' + this.id);
            // Initialize the plugin without toggling, then explicitly hide/show
            // so Bootstrap updates aria-expanded and the .collapsed class on
            // the trigger.
            $(this).collapse({ toggle: false })
                   .collapse(saved === 'collapsed' ? 'hide' : 'show');
        });
        $sections.on('shown.bs.collapse', function() {
            safeStorageSet('tuneCollapse:' + this.id, 'expanded');
        });
        $sections.on('hidden.bs.collapse', function() {
            safeStorageSet('tuneCollapse:' + this.id, 'collapsed');
        });
    }

    function paintTuningUI() {
        console.log('[tuning] paintTuningUI state=', JSON.stringify(tuningState));
        var painted = 0;

        // Skip writes to any input the user is actively interacting with,
        // so a server snapshot doesn't yank a slider/textbox back while
        // the user is mid-drag or mid-type. document.activeElement is the
        // currently focused element.
        function _shouldSkip(id) {
            var ae = document.activeElement;
            return ae && ae.id === id;
        }

        // HSV sliders
        ['center', 'edge'].forEach(function(group) {
            ['low', 'high'].forEach(function(bound) {
                var key = 'hsv_' + group + '_' + bound;
                var arr = tuningState[key];
                if (!Array.isArray(arr) || arr.length !== 3) {
                    console.warn('[tuning] paint skipped key=', key, 'value=', arr);
                    return;
                }
                ['h', 's', 'v'].forEach(function(ch, idx) {
                    var id = 'tune_hsv_' + group + '_' + ch + '_' + bound;
                    if (_shouldSkip(id)) return;
                    var $el = $('#' + id);
                    if (!$el.length) {
                        console.warn('[tuning] paint: missing element', id);
                        return;
                    }
                    $el.val(arr[idx]);
                    $('output[for="' + id + '"]').text(arr[idx]);
                    var $numEl = $('#' + id + '_num');
                    if ($numEl.length && !_shouldSkip(id + '_num')) $numEl.val(arr[idx]);
                    painted++;
                });
            });
        });

        // Scan + throttle + AI throttle/steering mult + half-track-width
        // (scalar sliders)
        ['scan_y', 'scan_height', 'throttle_min', 'throttle_max',
         'ai_throttle_mult', 'ai_steering_mult', 'half_track_width_px'].forEach(function(key) {
            var v = tuningState[key];
            if (v === null || v === undefined) {
                console.warn('[tuning] paint skipped key=', key, 'value=', v);
                return;
            }
            var id = 'tune_' + key;
            if (_shouldSkip(id)) return;
            $('#' + id).val(v);
            $('output[for="' + id + '"]').text(v);
            var $numEl = $('#' + id + '_num');
            if ($numEl.length && !_shouldSkip(id + '_num')) $numEl.val(v);
            painted++;
        });

        // PID numeric inputs. Floats like -0.01335 must reach <input
        // type="number"> as a string the browser accepts. step="0.00001"
        // means the browser is happy with up to 5 decimals; the literal
        // round-trip via .val() is fine.
        ['pid_p', 'pid_i', 'pid_d',
         'steering_left_pwm', 'steering_right_pwm',
         'throttle_forward_pwm', 'throttle_stopped_pwm', 'throttle_reverse_pwm',
         'steering_scale', 'throttle_scale'].forEach(function(key) {
            var v = tuningState[key];
            if (v === null || v === undefined || isNaN(v)) {
                console.warn('[tuning] paint skipped', key, 'value=', v);
                return;
            }
            var id = 'tune_' + key;
            if (_shouldSkip(id)) return;
            var $el = $('#' + id);
            if (!$el.length) {
                console.warn('[tuning] paint: missing element', id);
                return;
            }
            $el.val(v);
            painted++;
        });

        // Mode selector: paint the active radio (Bootstrap label needs the
        // 'active' class too) and toggle fieldset visibility. Skip if the
        // operator is currently clicking a radio (active focus).
        var mode = tuningState.line_follower_mode || 'center_line';
        var $radios = $('#tune_mode_group input[name="tune_mode"]');
        var focusedInGroup = $(document.activeElement)
            .closest('#tune_mode_group').length > 0;
        if ($radios.length && !focusedInGroup) {
            $radios.each(function() {
                var $r = $(this);
                var match = $r.val() === mode;
                $r.prop('checked', match);
                $r.closest('label').toggleClass('active', match);
            });
        }
        applyModeVisibility(mode);
        painted++;

        console.log('[tuning] paintTuningUI painted', painted, 'fields, mode=', mode);
    }

    function bindTuningSliders() {
        var sendDebounced = debounce(sendTuningPatch, 100);
        var bound = 0;

        // HSV sliders write into 3-element arrays.
        ['center', 'edge'].forEach(function(group) {
            ['low', 'high'].forEach(function(boundName) {
                var key = 'hsv_' + group + '_' + boundName;
                ['h', 's', 'v'].forEach(function(ch, idx) {
                    var id = 'tune_hsv_' + group + '_' + ch + '_' + boundName;
                    var $el = $('#' + id);
                    if (!$el.length) {
                        console.warn('[tuning] slider element missing:', id);
                        return;
                    }
                    bound++;
                    $el.on('input', function() {
                        console.log('[tuning] input fired on', id, 'value=', this.value);
                        var v = parseInt(this.value, 10);
                        if (isNaN(v)) return;
                        // Clone before mutating so we always send a fresh
                        // length-3 array (server validator requires it).
                        var arr = (tuningState[key] || [0, 0, 0]).slice();
                        arr[idx] = v;
                        tuningState[key] = arr;
                        $('output[for="' + id + '"]').text(v);
                        var patch = {};
                        patch[key] = arr;
                        sendDebounced(patch);
                    });
                    var $numEl = $('#' + id + '_num');
                    if ($numEl.length) {
                        $numEl.on('change', function() {
                            var v = parseInt(this.value, 10);
                            if (isNaN(v)) return;
                            var arr = (tuningState[key] || [0, 0, 0]).slice();
                            arr[idx] = v;
                            tuningState[key] = arr;
                            $('output[for="' + id + '"]').text(v);
                            $('#' + id).val(v);
                            var patch = {};
                            patch[key] = arr;
                            sendTuningPatch(patch);
                        });
                    }
                });
            });
        });
        console.log('[tuning] bindTuningSliders: bound', bound, 'HSV sliders');

        // Scan region + throttle limits + half-track-width (range sliders).
        var scalarBound = 0;
        [
            {id: 'tune_scan_y',              key: 'scan_y',              parse: function(s) { return parseInt(s, 10); }},
            {id: 'tune_scan_height',         key: 'scan_height',         parse: function(s) { return parseInt(s, 10); }},
            {id: 'tune_throttle_min',        key: 'throttle_min',        parse: parseFloat},
            {id: 'tune_throttle_max',        key: 'throttle_max',        parse: parseFloat},
            {id: 'tune_ai_throttle_mult',    key: 'ai_throttle_mult',    parse: parseFloat},
            {id: 'tune_ai_steering_mult',    key: 'ai_steering_mult',    parse: parseFloat},
            {id: 'tune_half_track_width_px', key: 'half_track_width_px', parse: function(s) { return parseInt(s, 10); }},
        ].forEach(function(s) {
            var $el = $('#' + s.id);
            if (!$el.length) {
                console.warn('[tuning] slider missing:', s.id);
                return;
            }
            scalarBound++;
            $el.on('input', function() {
                console.log('[tuning] input fired on', s.id, 'value=', this.value);
                var v = s.parse(this.value);
                if (isNaN(v)) return;
                tuningState[s.key] = v;
                $('output[for="' + s.id + '"]').text(v);
                var patch = {};
                patch[s.key] = v;
                sendDebounced(patch);
            });
            var $numEl = $('#' + s.id + '_num');
            if ($numEl.length) {
                $numEl.on('change', function() {
                    var v = s.parse(this.value);
                    if (isNaN(v)) return;
                    tuningState[s.key] = v;
                    $('output[for="' + s.id + '"]').text(v);
                    $('#' + s.id).val(v);
                    var patch = {};
                    patch[s.key] = v;
                    sendTuningPatch(patch);
                });
            }
        });
        console.log('[tuning] bound', scalarBound, 'scalar sliders');

        // Numeric inputs (PID + steering PWM endpoints) — commit on
        // every input (debounced) so the value reaches the car as you
        // type, not only on blur/Enter. 'change' also fires immediately
        // on Enter/Tab.
        var numBound = 0;
        var sendNumDebounced = debounce(sendTuningPatch, 200);
        var numericKeys = [
            {key: 'pid_p', parse: parseFloat},
            {key: 'pid_i', parse: parseFloat},
            {key: 'pid_d', parse: parseFloat},
            {key: 'steering_left_pwm',  parse: function(s) { return parseInt(s, 10); }},
            {key: 'steering_right_pwm', parse: function(s) { return parseInt(s, 10); }},
            {key: 'throttle_forward_pwm', parse: function(s) { return parseInt(s, 10); }},
            {key: 'throttle_stopped_pwm', parse: function(s) { return parseInt(s, 10); }},
            {key: 'throttle_reverse_pwm', parse: function(s) { return parseInt(s, 10); }},
            {key: 'steering_scale', parse: parseFloat},
            {key: 'throttle_scale', parse: parseFloat},
        ];
        numericKeys.forEach(function(spec) {
            var $el = $('#tune_' + spec.key);
            if (!$el.length) {
                console.warn('[tuning] numeric input missing:', 'tune_' + spec.key);
                return;
            }
            numBound++;
            var commit = function(immediate) {
                console.log('[tuning]', spec.key, 'value=', this.value, 'immediate=', !!immediate);
                var v = spec.parse(this.value);
                if (isNaN(v)) return;
                tuningState[spec.key] = v;
                var patch = {};
                patch[spec.key] = v;
                if (immediate) {
                    sendTuningPatch(patch);
                } else {
                    sendNumDebounced(patch);
                }
            };
            $el.on('input', function() { commit.call(this, false); });
            $el.on('change', function() { commit.call(this, true); });
        });
        console.log('[tuning] bound', numBound, 'numeric inputs');

        // Mode selector radios — send the chosen mode and locally apply
        // visibility immediately. Server will broadcast back the snapshot
        // for any other connected client, but our own paint is suppressed
        // by the focusedInGroup guard so the user's click doesn't bounce.
        $('#tune_mode_group input[name="tune_mode"]').on('change', function() {
            var mode = this.value;
            console.log('[tuning] mode change ->', mode);
            tuningState.line_follower_mode = mode;
            applyModeVisibility(mode);
            sendTuningPatch({line_follower_mode: mode});
        });
    }


    function bindInputModeToggle() {
        function applyMode(mode) {
            if (mode === 'number') {
                $('#tuning-body').addClass('tuning-mode-number');
                $('#toggle-input-mode-btn').text('↕ Sliders');
            } else {
                $('#tuning-body').removeClass('tuning-mode-number');
                $('#toggle-input-mode-btn').text('↕ Numbers');
            }
            safeStorageSet('tuneInputMode', mode);
        }
        var saved = safeStorageGet('tuneInputMode', 'slider');
        applyMode(saved);
        $('#toggle-input-mode-btn').on('click', function() {
            var current = $('#tuning-body').hasClass('tuning-mode-number') ? 'number' : 'slider';
            applyMode(current === 'number' ? 'slider' : 'number');
        });
    }

    function bindNipple(manager) {
      manager.on('start', function(evt, data) {
        state.tele.user.angle = 0
        state.tele.user.throttle = 0
        state.recording = true
        joystickLoopRunning=true;
        joystickLoop();

      }).on('end', function(evt, data) {
        joystickLoopRunning=false;
        brake()

      }).on('move', function(evt, data) {
        state.brakeOn = false;
        radian = data['angle']['radian']
        distance = data['distance']

        //console.log(data)
        state.tele.user.angle = Math.max(Math.min(Math.cos(radian)/70*distance, 1), -1)
        state.tele.user.throttle = limitedThrottle(Math.max(Math.min(Math.sin(radian)/70*distance , 1), -1))

        if (state.tele.user.throttle < .001) {
          state.tele.user.angle = 0
        }

      });
    }

    var updateUI = function() {
      $("#throttleInput").val(state.tele.user.throttle);
      $("#angleInput").val(state.tele.user.angle);
      $('#mode_select').val(state.driveMode);

      var throttlePercent = Math.round(Math.abs(state.tele.user.throttle) * 100) + '%';
      var steeringPercent = Math.round(Math.abs(state.tele.user.angle) * 100) + '%';
      var throttleRounded = state.tele.user.throttle.toFixed(2)
      var steeringRounded = state.tele.user.angle.toFixed(2)

      $('#throttle_label').html(throttleRounded);
      $('#steering_label').html(steeringRounded);

      if(state.tele.user.throttle < 0) {
        $('#throttle-bar-backward').css('width', throttlePercent).html(throttleRounded)
        $('#throttle-bar-forward').css('width', '0%').html('')
      }
      else if (state.tele.user.throttle > 0) {
        $('#throttle-bar-backward').css('width', '0%').html('')
        $('#throttle-bar-forward').css('width', throttlePercent).html(throttleRounded)
      }
      else {
        $('#throttle-bar-forward').css('width', '0%').html('')
        $('#throttle-bar-backward').css('width', '0%').html('')
      }

      if(state.tele.user.angle < 0) {
        $('#angle-bar-backward').css('width', steeringPercent).html(steeringRounded)
        $('#angle-bar-forward').css('width', '0%').html('')
      }
      else if (state.tele.user.angle > 0) {
        $('#angle-bar-backward').css('width', '0%').html('')
        $('#angle-bar-forward').css('width', steeringPercent).html(steeringRounded)
      }
      else {
        $('#angle-bar-forward').css('width', '0%').html('')
        $('#angle-bar-backward').css('width', '0%').html('')
      }

      if (state.recording) {
        $('#record_button')
          .html('Stop Recording (r)')
          .removeClass('btn-info')
          .addClass('btn-warning').end()
      } else {
        $('#record_button')
          .html('Start Recording (r)')
          .removeClass('btn-warning')
          .addClass('btn-info').end()
      }

      if (state.brakeOn) {
        $('#brake_button')
          .html('Start Vehicle')
          .removeClass('btn-danger')
          .addClass('btn-success').end()
      } else {
        $('#brake_button')
          .html('Stop Vehicle')
          .removeClass('btn-success')
          .addClass('btn-danger').end()
      }

      if(deviceHasOrientation) {
        $('#tilt-toggle').removeAttr("disabled")
        $('#tilt').removeAttr("disabled")
      } else {
        $('#tilt-toggle').attr("disabled", "disabled");
        $('#tilt').prop("disabled", true);
      }

      if(hasGamepad) {
        $('#gamepad-toggle').removeAttr("disabled")
        $('#gamepad').removeAttr("disabled")
      } else {
        $('#gamepad-toggle').attr("disabled", "disabled");
        $('#gamepad').prop("disabled", true);
      }

      if (state.controlMode == "joystick") {
        $('#joystick_outer').show();
        $('#joystick-toggle').addClass("active");
        $('#joystick').attr("checked", "checked")
      } else {
        $('#joystick_outer').hide();
        $('#joystick-toggle').removeClass("active");
        $('#joystick').removeAttr("checked");
      }

      if (state.controlMode == "tilt") {
        $('#tilt-toggle').addClass("active");
        $('#tilt').attr("checked", "checked");
      } else {
        $('#tilt-toggle').removeClass("active");
        $('#tilt').removeAttr("checked")
      }

      //drawLine(state.tele.user.angle, state.tele.user.throttle)
    };

    const ALL_POST_FIELDS = ['angle', 'throttle', 'drive_mode', 'recording', 'buttons'];

    //
    // Set any changed properties to the server
    // via the websocket connection
    //
    var postDrive = function(fields=[]) {

        if(fields.length === 0) {
            fields = ALL_POST_FIELDS;
        }

        let data = {}
        fields.forEach(field => {
            switch (field) {
                case 'angle': data['angle'] = state.tele.user.angle; break;
                case 'throttle': data['throttle'] = state.tele.user.throttle; break;
                case 'drive_mode': data['drive_mode'] = state.driveMode; break;
                case 'recording': data['recording'] = state.recording; break;
                case 'buttons': data['buttons'] = state.buttons; break;
                default: console.log(`Unexpected post field: '${field}'`); break;
            }
        });
        if(data) {
            let json_data = JSON.stringify(data);
            console.log(`Posting ${json_data}`);
            socket.send(json_data)
            updateUI()
        }
    };

    var applyDeadzone = function(number, threshold){
       percentage = (Math.abs(number) - threshold) / (1 - threshold);

       if(percentage < 0)
          percentage = 0;

       return percentage * (number > 0 ? 1 : -1);
    }



    function gamePadLoop() {
      setTimeout(gamePadLoop,100);

      if (state.controlMode != "gamepad") {
        return;
      }

      var gamepads = navigator.getGamepads();

      for (var i = 0; i < gamepads.length; ++i)
        {
          var pad = gamepads[i];
          // some pads are NULL I think.. some aren't.. use one that isn't null
          if (pad && pad.timestamp!=0)
          {

            var joystickX = applyDeadzone(pad.axes[2], 0.05);

            var joystickY = applyDeadzone(pad.axes[1], 0.15);

            state.tele.user.angle = joystickX;
            state.tele.user.throttle = limitedThrottle((joystickY * -1));

            if (state.tele.user.throttle == 0 && state.tele.user.throttle == 0) {
              state.brakeOn = true;
            } else {
              state.brakeOn = false;
            }

            if (state.tele.user.throttle != 0) {
              state.recording = true;
            } else {
              state.recording = false;
            }

            postDrive()

          }
            // todo; simple demo of displaying pad.axes and pad.buttons
        }
      }


    // Send control updates to the server every .1 seconds.
    function joystickLoop () {
       setTimeout(function () {
            postDrive()

          if (joystickLoopRunning && state.controlMode == "joystick") {
             joystickLoop();
          }
       }, 100)
    }

    // Control throttle and steering with device orientation
    function handleOrientation(event) {

      var alpha = event.alpha;
      var beta = event.beta;
      var gamma = event.gamma;

      if (beta == null || gamma == null) {
        deviceHasOrientation = false;
        state.controlMode = "joystick";
        console.log("Invalid device orientation values, switched to joystick mode.")
      } else {
        deviceHasOrientation = true;
        console.log("device has valid orientation values")
      }

      updateUI();

      if(state.controlMode != "tilt" || !deviceHasOrientation || state.brakeOn){
        return;
      }

      if(!initialGamma && gamma) {
        initialGamma = gamma;
      }

      var newThrottle = gammaToThrottle(gamma);
      var newAngle = betaToSteering(beta, gamma);

      // prevent unexpected switch between full forward and full reverse
      // when device is parallel to ground
      if (state.tele.user.throttle > 0.9 && newThrottle <= 0) {
        newThrottle = 1.0
      }

      if (state.tele.user.throttle < -0.9 && newThrottle >= 0) {
        newThrottle = -1.0
      }

      state.tele.user.throttle = limitedThrottle(newThrottle);
      state.tele.user.angle = newAngle;
    }

    function deviceOrientationLoop () {
       setTimeout(function () {
          if(!state.brakeOn){
            postDrive()
          }

          if (state.controlMode == "tilt") {
            deviceOrientationLoop();
          }
       }, 100)
    }

    var throttleUp = function(){
      state.tele.user.throttle = limitedThrottle(Math.min(state.tele.user.throttle + .05, 1));
      postDrive()
    };

    var throttleDown = function(){
      state.tele.user.throttle = limitedThrottle(Math.max(state.tele.user.throttle - .05, -1));
      postDrive()
    };

    var angleLeft = function(){
      state.tele.user.angle = Math.max(state.tele.user.angle - .1, -1)
      postDrive()
    };

    var angleRight = function(){
      state.tele.user.angle = Math.min(state.tele.user.angle + .1, 1)
      postDrive()
    };

    var updateDriveMode = function(mode){
      state.driveMode = mode;
      postDrive(["drive_mode"])
    };

    var toggleDriveMode = function() {
      switch(state.driveMode) {
        case "user": {
            updateDriveMode("local_angle");
            break;
        }
        case "local_angle": {
            updateDriveMode("local");
            break;
        }
        default: {
            updateDriveMode("user");
            break;
        }
      }
    }

    var toggleRecording = function(){
      state.recording = !state.recording
      postDrive(['recording']);
    };

    var toggleBrake = function(){
      state.brakeOn = !state.brakeOn;
      initialGamma = null;

      if (state.brakeOn) {
        brake();
      }
    };

    var brake = function(i){
          console.log('post drive: ' + i)
          state.tele.user.angle = 0
          state.tele.user.throttle = 0
          state.recording = false
          state.driveMode = 'user';
          postDrive()

      i++
      if (i < 5) {
        setTimeout(function () {
          console.log('calling brake:' + i)
          brake(i);
        }, 500)
      };

      state.brakeOn = true;
      updateUI();
    };

    var limitedThrottle = function(newThrottle){
      var limitedThrottle = 0;

      if (newThrottle > 0) {
        limitedThrottle = Math.min(state.maxThrottle, newThrottle);
      }

      if (newThrottle < 0) {
        limitedThrottle = Math.max((state.maxThrottle * -1), newThrottle);
      }

      if (state.throttleMode == 'constant') {
        limitedThrottle = state.maxThrottle;
      }

      return limitedThrottle;
    }


    // var drawLine = function(angle, throttle) {
    //
    //   throttleConstant = 100
    //   throttle = throttle * throttleConstant
    //   angleSign = Math.sign(angle)
    //   angle = toRadians(Math.abs(angle*90))
    //
    //   var canvas = document.getElementById("angleView"),
    //   context = canvas.getContext('2d');
    //   context.clearRect(0, 0, canvas.width, canvas.height);
    //
    //   base={'x':canvas.width/2, 'y':canvas.height}
    //
    //   pointX = Math.sin(angle) * throttle * angleSign
    //   pointY = Math.cos(angle) * throttle
    //   xPoint = {'x': pointX + base.x, 'y': base.y - pointY}
    //
    //   context.beginPath();
    //   context.moveTo(base.x, base.y);
    //   context.lineTo(xPoint.x, xPoint.y);
    //   context.lineWidth = 5;
    //   context.strokeStyle = '#ff0000';
    //   context.stroke();
    //   context.closePath();
    //
    // };

    var betaToSteering = function(beta, gamma) {
      const deadZone = 5;
      var angle = 0.0;
      var outsideDeadZone = false;
      var controlDirection = (Math.sign(initialGamma) * -1)

      //max steering angle at device 35º tilt
      var fullLeft = -35.0;
      var fullRight = 35.0;

      //handle beta 90 to 180 discontinuous transition at gamma 90
      if (beta > 90) {
        beta = (beta - 180) * Math.sign(gamma * -1) * controlDirection
      } else if (beta < -90) {
        beta = (beta + 180) * Math.sign(gamma * -1) * controlDirection
      }

      // set the deadzone for neutral sterring
      if (Math.abs(beta) > 90) {
        outsideDeadZone = Math.abs(beta) < 180 - deadZone;
      }
      else {
        outsideDeadZone = Math.abs(beta) > deadZone;
      }

      if (outsideDeadZone && beta < -90.0) {
        angle = remap(beta, fullLeft, (-180.0 + deadZone), -1.0, 0.0);
      }
      else if (outsideDeadZone && beta > 90.0) {
        angle = remap(beta, (180.0 - deadZone), fullRight, 0.0, 1.0);
      }
      else if (outsideDeadZone && beta < 0.0) {
        angle = remap(beta, fullLeft, 0.0 - deadZone, -1.0, 0);
      }
      else if (outsideDeadZone && beta > 0.0) {
        angle = remap(beta, 0.0 + deadZone, fullRight, 0.0, 1.0);
      }

      // set full turn if abs(angle) > 1
      if (angle < -1) {
        angle = -1;
      } else if (angle > 1) {
        angle = 1;
      }

      return angle * controlDirection;
    };

    var gammaToThrottle = function(gamma) {
      var throttle = 0.0;
      var gamma180 = gamma + 90;
      var initialGamma180 = initialGamma + 90;
      var controlDirection = (Math.sign(initialGamma) * -1);

      // 10 degree deadzone around the initial position
      // 45 degrees of motion for forward and reverse
      var minForward = Math.min((initialGamma180 + (5 * controlDirection)), (initialGamma180 + (50 * controlDirection)));
      var maxForward = Math.max((initialGamma180 + (5 * controlDirection)), (initialGamma180 + (50 * controlDirection)));
      var minReverse = Math.min((initialGamma180 - (50 * controlDirection)), (initialGamma180 - (5 * controlDirection)));
      var maxReverse = Math.max((initialGamma180 - (50 * controlDirection)), (initialGamma180 - (5 * controlDirection)));

      //constrain control input ranges to 0..180 continuous range
      minForward = Math.max(minForward, 0);
      maxForward = Math.min(maxForward, 180);
      minReverse = Math.max(minReverse, 0);
      maxReverse = Math.min(maxReverse, 180);

      if(gamma180 > minForward && gamma180 < maxForward) {
        // gamma in forward range
        if (controlDirection == -1) {
          throttle = remap(gamma180, minForward, maxForward, 1.0, 0.0);
        } else {
          throttle = remap(gamma180, minForward, maxForward, 0.0, 1.0);
        }
      } else if (gamma180 > minReverse && gamma180 < maxReverse) {
        // gamma in reverse range
        if (controlDirection == -1) {
          throttle = remap(gamma180, minReverse, maxReverse, 0.0, -1.0);
        } else  {
          throttle = remap(gamma180, minReverse, maxReverse, -1.0, 0.0);
        }
      }

      return throttle;
    };

}();


function toRadians (angle) {
  return angle * (Math.PI / 180);
}

function remap( x, oMin, oMax, nMin, nMax ){
  //range check
  if (oMin == oMax){
      console.log("Warning: Zero input range");
      return None;
  };

  if (nMin == nMax){
      console.log("Warning: Zero output range");
      return None
  }

  //check reversed input range
  var reverseInput = false;
  oldMin = Math.min( oMin, oMax );
  oldMax = Math.max( oMin, oMax );
  if (oldMin != oMin){
      reverseInput = true;
  }

  //check reversed output range
  var reverseOutput = false;
  newMin = Math.min( nMin, nMax )
  newMax = Math.max( nMin, nMax )
  if (newMin != nMin){
      reverseOutput = true;
  };

  var portion = (x-oldMin)*(newMax-newMin)/(oldMax-oldMin)
  if (reverseInput){
      portion = (oldMax-x)*(newMax-newMin)/(oldMax-oldMin);
  };

  var result = portion + newMin
  if (reverseOutput){
      result = newMax - portion;
  }

return result;
}

