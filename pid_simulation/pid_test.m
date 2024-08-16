
% Parameters
Kp = 0.981; % Proportional gain
Ki = 0.193; % Integral gain
Kd = 0; % Derivative gain


% Transfer function of the linearized system
s = tf('s');
PID = Kp + Ki/s + Kd*s;
linearized_system = PID / s;


% Plot the step response
figure;
step(linearized_system);
title('Step Response of the Linearized System');
grid on;

% Bode plot to analyze stability margins
figure;
margin(linearized_system);
title('Bode Plot of the Linearized System');
grid on;

% Nyquist plot to analyze stability
figure;
nyquist(linearized_system);
title('Nyquist Plot of the Linearized System');
grid on;

