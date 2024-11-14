function calculateRevisitTime(access)
    intervals = accessIntervals(access);
    startTimes = intervals.StartTime; % Hent starttidspunktene
    startTimesNum = datenum(startTimes); % Konverter til numeriske verdier
    revisitTimes = diff(startTimesNum) * 24 * 60; % Beregn forskjellene og konverter til minutter
    
    avgRevisitTime = mean(revisitTimes); % Gjennomsnittlig revisit-tid i minutter
    maxRevisitTime = max(revisitTimes); % Maksimal revisit-tid i minutter
    
    % Konverter til HH:MM:SS format
    avgRevisitDuration = duration(0, avgRevisitTime, 0, 'Format', 'hh:mm:ss');
    maxRevisitDuration = duration(0, maxRevisitTime, 0, 'Format', 'hh:mm:ss');
    
    fprintf('Gjennomsnittlig revisit-tid: %s\n', avgRevisitDuration);
    fprintf('Maksimal revisit-tid: %s\n', maxRevisitDuration);
end