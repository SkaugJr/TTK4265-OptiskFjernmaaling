startTime = datetime(2024,03,20,0,0,0);
stopTime = startTime + days(7); % Øk til 30 dager for mere presis revisit-tid
sampleTime = 60;        
sc = satelliteScenario(startTime, stopTime, sampleTime);

Sat400 = satellite(sc,"TLE_400.txt","Name","400 km");
cam400 = conicalSensor(Sat400,"Name", Sat400.Name + " Camera","MaxViewAngle",20);
fov400 = fieldOfView(cam400([cam400.Name] == "400 km Camera"));

Sat500 = satellite(sc,"TLE_500.txt","Name","500 km");
cam500 = conicalSensor(Sat500,"Name", Sat500.Name + " Camera","MaxViewAngle",20);
fov500 = fieldOfView(cam500([cam500.Name] == "500 km Camera"));

Sat600 = satellite(sc,"TLE_600.txt","Name","600 km");
cam600 = conicalSensor(Sat600,"Name", Sat600.Name + " Camera","MaxViewAngle",20);
fov600 = fieldOfView(cam600([cam600.Name] == "600 km Camera"));

% Basestasjon på Svalbard
svalbard = groundStation(sc, 78.2298, 15.4078, "Name", "Svalbard");

% Overvåkningsområde
Oslo = groundStation(sc, 59.9139, 10.7522, "Name",  "Oslo","MinElevationAngle",45);

pointAt(Sat400,Oslo);
pointAt(Sat500,Oslo);
pointAt(Sat600,Oslo);

% Beregn revisit-tid
access400 = access(Sat400, Oslo);
access500 = access(Sat500, Oslo);
access600 = access(Sat600, Oslo);


% Kalkuler revisit-tid for hver satellitt
fprintf('400 km satellitt:\n');
calculateRevisitTime(access400);

fprintf('\n500 km satellitt:\n');
calculateRevisitTime(access500);

fprintf('\n600 km satellitt:\n');
calculateRevisitTime(access600);

% Beregn downlink-tid
downlink400 = access(Sat400, svalbard);
downlink500 = access(Sat500, svalbard);
downlink600 = access(Sat600, svalbard);

% Kalkuler gjennomsnittlig downlink tid per periode og total tid per dag for hver satellitt
fprintf('400 km satellitt:\n');
calculateAverageAccessTime(downlink400, startTime, stopTime);

fprintf('\n500 km satellitt:\n');
calculateAverageAccessTime(downlink500, startTime, stopTime);

fprintf('\n600 km satellitt:\n');
calculateAverageAccessTime(downlink600, startTime, stopTime);

% Play the satellite scenario visualization
play(sc)
