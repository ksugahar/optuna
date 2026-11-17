function [value, mcp] = version()
%VERSION Return the upstream Optuna compatibility version.
%   MATLAB identifiers cannot begin with two underscores, so this function
%   is the language-equivalent surface for both optuna.__version__ and
%   optuna.version.__version__.

matlabRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
packageRoot = fileparts(matlabRoot);
pinPath = fullfile(packageRoot, 'upstream.json');
if ~isfile(pinPath)
    pinPath = fullfile(packageRoot, 'src', 'radia_optuna', 'upstream.json');
end
pin = jsondecode(fileread(pinPath));
value = string(pin.optuna);
mcp = string(pin.optuna_mcp);
end
