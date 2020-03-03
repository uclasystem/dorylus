function getLocation(name) {
    if (name.charAt(name.indexOf("_") + 1) == "L")
        return "Device" + name.substring(name.indexOf("Id") + 2, name.indexOf("Ly"))
    if (name.charAt(0)=="L")
        return "Local(λ)"
    if (name.charAt(0)=="C")
        return "Local(CPU)"
    return "Local(Unknown)"
}
function getID(name) {
    return name.substring(name.indexOf("Id") + 2, name.indexOf("Ly"))
}

function getType(name) {
    if (getLocation(name).charAt(0) == 'L'){
        return name.replace("LAMBDA_","").replace("CPU_","")
    }
    else {
        if (name.includes("LNREQ"))
            return "R"
        if (name.includes("LC"))
            return "Comp"
        if (name.includes("LNSend"))
            return "S"
    }
    return ""
}

function getTimelineList() {
    const Http = new XMLHttpRequest();
    const url = window.location.href + '/datalist';
    Http.onreadystatechange = function () {
        if (Http.readyState == 4 && Http.status == 200) {
            var respData = JSON.parse(Http.responseText) || {};
            const sel = document.getElementById("timelines")
            respData.sort()
            sel.innerHTML = ""
            for (var i = 0; i < respData.length; i++) {
                var new_option = document.createElement("option");
                new_option.text = respData[i];
                sel.appendChild(new_option)
            }
        }
    };
    Http.open("GET", url);
    Http.send();
}


google.charts.load('current', {
    'packages': ['timeline']
});


function drawChart(filename = "default") {
    var container = document.getElementById('timeline');
    var chart = new google.visualization.Timeline(container);
    var dataTable = new google.visualization.DataTable();

    dataTable.addColumn({
        type: 'string',
        id: 'location'
    });
    dataTable.addColumn({
        type: 'string',
        id: 'Function'
    });

    dataTable.addColumn({
        type: 'date',
        id: 'Start'
    });
    dataTable.addColumn({
        type: 'date',
        id: 'End'
    });
    var options = {
        colors: ['#cbb69d', '#603913', '#c69c6e'],
        timeline: { showRowLabels: true },
        tooltip: { isHtml: true }
    };
    const Http = new XMLHttpRequest();
    const url = window.location.href + '/' + filename;
    Http.onreadystatechange = function () {
        if (Http.readyState == 4 && Http.status == 200) {
            var respData = JSON.parse(Http.responseText) || {};
            // console.log(respData)
            var rows = []
            for (var i = 0; i < respData.length; i++) {
                // console.log(getLocation(respData[i][0]))
                rows.push(
                    [getLocation(respData[i][0]),
                    getType(respData[i][0]),
                    new Date(parseInt(respData[i][1])),
                    new Date(parseInt(respData[i][2]))])
            }
            dataTable.addRows(rows);
            // console.log(rows)
            chart.draw(dataTable, options);
        }
    };
    Http.open("GET", url);
    Http.send();
}
window.addEventListener("load", () => {
    getTimelineList();
    const selection = document.getElementById("timelines");
    selection.style.height = window.height;
    selection.addEventListener("change", () => {
        drawChart(selection.options[selection.selectedIndex].value)
    });
})
google.charts.setOnLoadCallback(drawChart);