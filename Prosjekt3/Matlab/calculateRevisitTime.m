function calculateRevisitTime(access)
    intervals = accessIntervals(access);
    startTimes = intervals.StartTime; % Hent starttidspunktene
    startTimesNum = datenum(startTimes); % Konverter til numeriske verdier
    revisitTimes = diff(startTimesNum); % Beregn forskjellene
    avgRevisitTime = mean(revisitTimes) * 24 * 60; % Konverter fra dager til minutter
    maxRevisitTime = max(revisitTimes) * 24 * 60; % Konverter fra dager til minutter
    fprintf('Gjennomsnittlig revisit-tid: %.2f minutter\n', avgRevisitTime);
    fprintf('Maksimal revisit-tid: %.2f minutter\n', maxRevisitTime);
end