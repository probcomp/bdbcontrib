angular.module('bdbServices', ['ngResource'])
    .factory('Table', function($resource) {
        return $resource('/api/v1/table/:tableName', {}, {
            query: {method:'GET', params:{tableName:''}, isArray:true}
        });
    })
    .factory('BQL', function($resource) {
        return $resource('/api/v1/bql/:query', {}, {
            query: {method:'GET', params:{query:''}, isArray:true}
        });
    });


angular.module('bdbRestExampleApp', ['ngResource', 'bdbServices'])
    .controller('TablesListCtrl', function($scope, Table) {
        $scope.tables = Table.query();
    })
    .controller('SimulateCtrl', function($scope, BQL) {
        $scope.query = "select * from satellites";
        var f = BQL.query({query: $scope.query});
        f.$promise.then(function(results){
            $scope.data = results;
        });
        $scope.setAP = function(a, p) {
            $scope.apogee = a;
            $scope.perigee = p;
            var sim = BQL.query({query: "simulate Country_of_Operator from satellites_cc given Apogee_km = " + a + " and Perigee_km = " + p + " limit 100"});
            sim.$promise.then(function (results) {
                $scope.sim = results;
            });
        };
    })

.directive('bdbScatter', function () {

  // constants
  var margin = 20,
    width = 960,
    height = 500 - .5 - margin,
    color = d3.interpolateRgb("#f77", "#77f");

  return {
    restrict: 'E',
    scope: {
      val: '=',
      set: '='
    },
    link: function (scope, element, attrs) {

      // set up initial svg object
      var vis = d3.select(element[0])
        .append("svg")
          .attr("width", width)
          .attr("height", height + margin + 100);

      scope.$watch('val', function (newVal, oldVal) {

        // clear the elements inside of the directive
        vis.selectAll('*').remove();

        // if 'val' is undefined, exit
        if (!newVal || newVal.length == 0) {
          return;
        }

          /*
           * value accessor - returns the value to encode for a given data object.
           * scale - maps value to a visual display encoding, such as a pixel position.
           * map function - maps from data value to display value
           * axis - sets up axis
           */

          // setup x
          var xValue = function(d) { return d.Apogee_km;}, // data -> value
              xScale = d3.scale.log().range([0, width]), // value -> display
              xMap = function(d) { return xScale(xValue(d));}, // data -> display
              xAxis = d3.svg.axis().scale(xScale).orient("bottom");

          // setup y
          var yValue = function(d) { return d.Perigee_km;}, // data -> value
              yScale = d3.scale.log().range([height, 0]), // value -> display
              yMap = function(d) { return yScale(yValue(d));}, // data -> display
              yAxis = d3.svg.axis().scale(yScale).orient("left");

          // setup fill color
          var colorMap = d3.nest().key(function (d) { return d.Country_of_Operator; })
                  .rollup(function(v) { return v.length; })
                  .entries(newVal);
          colorMap.sort(function (a,b) { return b.values - a.values;});
          colorMap = colorMap.slice(0,10);
          colorMap = d3.map(colorMap, function (e) { return e.key; });
          var cValue = function(d) {
              var c = d.Country_of_Operator;
              return colorMap.has(c) ? c : 'Other';
          },
              color = d3.scale.category10();

          // add the tooltip area to the webpage
          var tooltip = vis.append("div")
                  .attr("class", "mytooltip")
                  .style("opacity", 0);

          var data = newVal;

          // don't want dots overlapping axis, so add in buffer to data domain
          xScale.domain([d3.min(data, xValue)-1, d3.max(data, xValue)+1]);
          yScale.domain([d3.min(data, yValue)-1, d3.max(data, yValue)+1]);

          vis.on("click", function() {
              var point = d3.mouse(this);
              var x = xScale.invert(point[0]);
              var y = yScale.invert(point[1]);
              console.log("Click ", x, y);
              scope.$apply(function() {
                  scope.set(x, y);
                  console.log("Applied");
              });
          });

          // x-axis
          vis.append("g")
              .attr("class", "x axis")
              .attr("transform", "translate(0," + height + ")")
              .call(xAxis)
              .append("text")
              .attr("class", "label")
              .attr("x", width)
              .attr("y", -6)
              .style("text-anchor", "end")
              .text("Apogee");

          // y-axis
          vis.append("g")
              .attr("class", "y axis")
              .call(yAxis)
              .append("text")
              .attr("class", "label")
              .attr("transform", "rotate(-90)")
              .attr("y", 6)
              .attr("dy", ".71em")
              .style("text-anchor", "end")
              .text("Perigee");

          // draw dots
          vis.selectAll(".dot")
              .data(data)
              .enter().append("circle")
              .attr("class", "dot")
              .attr("r", 3.5)
              .attr("cx", xMap)
              .attr("cy", yMap)
              .style("fill", function(d) { return color(cValue(d));})
              .on("mouseover", function(d) {
                  tooltip.transition()
                      .duration(200)
                      .style("opacity", .9);
                  tooltip.html(d.Name + "<br/> (" + xValue(d)
	                       + ", " + yValue(d) + ")")
                      .style("left", (d3.event.pageX + 5) + "px")
                      .style("top", (d3.event.pageY - 28) + "px");
              })
              .on("mouseout", function(d) {
                  tooltip.transition()
                      .duration(500)
                      .style("opacity", 0);
              });

          // draw legend
          var legend = vis.selectAll(".legend")
                  .data(color.domain())
                  .enter().append("g")
                  .attr("class", "legend")
                  .attr("transform", function(d, i) { return "translate(0," + i * 20 + ")"; });

          // draw legend colored rectangles
          legend.append("rect")
              .attr("x", width - 18)
              .attr("width", 18)
              .attr("height", 18)
              .style("fill", color);

          // draw legend text
          legend.append("text")
              .attr("x", width - 24)
              .attr("y", 9)
              .attr("dy", ".35em")
              .style("text-anchor", "end")
              .text(function(d) { return d;});
      });
    }
  };
});
