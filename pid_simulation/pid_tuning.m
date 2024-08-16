% Définir le système
s = tf('s');
sys = 1/s;

% Spécifier les options pour pidtune
% Par exemple, augmenter la bande passante cible pour un temps de réponse 
% plus rapide.
% Fréquence de croisement plus élevée pour une réponse plus rapide
% Marge de phase souhaitée pour une meilleure stabilité
% Prioriser la performance pour une réponse rapide
options = pidtuneOptions('CrossoverFrequency', 10, ...  
                         'PhaseMargin', 60, ...         
                         'DesignFocus', 'reference-tracking'); 

% Tuner le contrôleur PID
[C_pid, info] = pidtune(sys, 'PID', options);

% Afficher les paramètres du PID
Kp = C_pid.Kp;
Ki = C_pid.Ki;
Kd = C_pid.Kd;
disp('Gains du PID:');
disp(['Kp = ', num2str(Kp)]);
disp(['Ki = ', num2str(Ki)]);
disp(['Kd = ', num2str(Kd)]);

% Afficher les informations sur la bande passante et la marge de phase
disp('Informations de pidtune:');
disp(info);

% Créer une boucle fermée avec le contrôleur PID
closed_loop_sys = feedback(C_pid * sys, 1);

% Réponse en échelon
figure;
step(closed_loop_sys);
title('Réponse en échelon du système en boucle fermée avec PID');
grid on;

% Bode plot pour analyser les marges de gain et de phase
figure;
margin(closed_loop_sys);
title('Bode Plot du système en boucle fermée avec PID');
grid on;

% Nyquist plot pour analyser la stabilité
figure;
nyquist(closed_loop_sys);
title('Nyquist Plot du système en boucle fermée avec PID');
grid on;

